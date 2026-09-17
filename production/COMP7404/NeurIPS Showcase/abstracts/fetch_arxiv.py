# -*- coding: utf-8 -*-
"""
按 oral / spotlight 论文标题到 arXiv 抓取 abstract，并记录 arXiv 链接。

与 fetch_abstracts.py 的区别：
  - arXiv 标题检索为主（可拿到 abstract + arXiv id/url，供页面标题跳转）
  - arXiv 未命中时回退 OpenReview 检索（仅补 abstract）
  - 结果写入独立进度文件 arxiv_progress.jsonl，字段含 arxiv_id / arxiv_url

断点续跑：每篇处理完立即 append，重启自动跳过已完成项。

用法：
  python fetch_arxiv.py --max 20      # 先跑少量验证
  python fetch_arxiv.py               # 全部待处理（可中断续跑）
  python fetch_arxiv.py --retry-miss  # 重跑此前失败项
  python fetch_arxiv.py --report      # 只输出进度报告
"""

import argparse
import csv
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = r"D:\HKU\course\COMP7404\NeurIPS Showcase"
CSV_IN = os.path.join(BASE, "build", "NeurIPS2025_oral_spotlight_论文分类清单.csv")
OUTDIR = os.path.join(BASE, "abstracts")
PROGRESS = os.path.join(OUTDIR, "arxiv_progress.jsonl")

UA = "Mozilla/5.0 (compatible; course-abstract-fetcher/1.0)"
ARXIV_DELAY = 3.0
OR_DELAY = 1.2
TIMEOUT = 30
MAX_RETRY = 4
ARX_MIN_SCORE = 0.60

socket.setdefaulttimeout(TIMEOUT)

STOP = set("""a an the of for and or on in to with via from under over towards toward
is are be been being it its as at by we our us their this that these those can could
should would may might will not no""".split())

LATEX_MAP = {
    "ell": "l", "phi": "phi", "psi": "psi", "alpha": "alpha", "beta": "beta",
    "gamma": "gamma", "delta": "delta", "theta": "theta", "lambda": "lambda",
    "mu": "mu", "sigma": "sigma", "tau": "tau", "pi": "pi", "rho": "rho",
    "omega": "omega", "mathcal": "", "boldsymbol": "", "tilde": "", "hat": "",
    "text": "", "mathrm": "", "mathbf": "", "frac": "", "sqrt": "", "cdot": "",
    "times": "x", "approx": "", "leq": "", "geq": "", "infty": "",
}


def norm_tokens(title):
    t = (title or "").lower()
    t = re.sub(r"\\([a-zA-Z]+)", lambda m: " " + LATEX_MAP.get(m.group(1), m.group(1)) + " ", t)
    t = t.replace("$", " ")
    t = re.sub(r"[_{}~^\\]", " ", t)
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return [w for w in t.split() if len(w) > 1 and w not in STOP]


def search_query_variants(title):
    t = re.sub(r"\\([a-zA-Z]+)", lambda m: " " + LATEX_MAP.get(m.group(1), m.group(1)) + " ",
               (title or "").lower())
    t = t.replace("$", " ").replace("&", " and ")
    t = re.sub(r"[_{}~^\\]", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    toks = [w for w in t.split() if len(w) > 1 and w not in STOP]
    variants = []
    if toks:
        variants.append(" ".join(toks))
        variants.append(" ".join(toks[:12]))
        if len(toks) > 6:
            variants.append(" ".join(toks[:6]))
    seen, out = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def arxiv_query_variants(title):
    """arXiv 的 ti: 是短语检索，必须保留全部词且顺序不变，只把标点归一为空格。
    去停用词会破坏短语匹配（实测会 0 命中），因此这里单独生成查询。"""
    t = re.sub(r"\\([a-zA-Z]+)", lambda m: " " + LATEX_MAP.get(m.group(1), m.group(1)) + " ",
               (title or "").lower())
    t = t.replace("$", " ").replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    toks = t.split()
    variants = []
    if toks:
        variants.append(" ".join(toks))          # 全标题短语（首选，命中率最高）
        variants.append(" ".join(toks[:8]))       # 前 8 词
        if len(toks) > 5:
            variants.append(" ".join(toks[:5]))   # 前 5 词（最宽松）
    seen, out = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def score(query_toks, cand_title):
    ct = norm_tokens(cand_title)
    if not query_toks or not ct:
        return 0.0
    qs, cs = set(query_toks), set(ct)
    cover = len(qs & cs) / len(qs)
    ratio = min(len(qs), len(cs)) / max(len(qs), len(cs))
    return round(cover * 0.85 + ratio * 0.15, 4)


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/json, text/xml, */*"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", "replace")


def http_get_retry(url):
    last = None
    for i in range(MAX_RETRY):
        try:
            return http_get(url), None
        except urllib.error.HTTPError as e:
            last = "HTTP %s" % e.code
            if e.code in (400, 403, 404, 410):
                return None, last
            time.sleep(2.0 * (i + 1))
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:120])
            time.sleep(2.0 * (i + 1))
    return None, last


# ---------------------------------------------------------------- arXiv

ARX_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
ARX_ID = re.compile(r"<id>\s*(.*?)\s*</id>", re.S)
ARX_T = re.compile(r"<title>(.*?)</title>", re.S)
ARX_S = re.compile(r"<summary>(.*?)</summary>", re.S)
ARX_PUB = re.compile(r"<published>\s*(.*?)\s*</published>", re.S)


def unescape_xml(s):
    return (s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"').replace("&apos;", "'").replace("&#39;", "'"))


def fetch_arxiv(title):
    """返回 (abstract, arxiv_id, arxiv_url, matched_title, published, score, err)"""
    qt = norm_tokens(title)
    last_err = "no entries"
    for q in arxiv_query_variants(title):
        query = urllib.parse.quote('ti:"%s"' % q)
        url = "http://export.arxiv.org/api/query?search_query=%s&max_results=8" % query
        body, err = http_get_retry(url)
        if body is None:
            last_err = err or "arxiv empty"
            continue
        entries = ARX_ENTRY.findall(body)
        if not entries:
            last_err = "no entries (term=%r)" % q[:40]
            continue
        best = None
        for e in entries:
            mt, ms, mi = ARX_T.search(e), ARX_S.search(e), ARX_ID.search(e)
            if not mt or not ms or not mi:
                continue
            ct = unescape_xml(mt.group(1)).strip()
            ab = unescape_xml(ms.group(1)).strip()
            ab = re.sub(r"\s+", " ", ab)
            aid = mi.group(1).strip()
            mp = ARX_PUB.search(e)
            pub = mp.group(1).strip() if mp else ""
            s = score(qt, ct)
            if best is None or s > best[0]:
                best = (s, ab, aid, ct, pub)
        if best is None:
            last_err = "entries without abstract"
            continue
        s, ab, aid, ct, pub = best
        if s < ARX_MIN_SCORE:
            last_err = "low score %.2f (best=%s)" % (s, ct[:70])
            continue
        m = re.search(r"abs/([0-9.]+)", aid)
        short = m.group(1) if m else aid
        return ab, short, "https://arxiv.org/abs/%s" % short, ct, pub, s, None
    return "", "", "", "", "", 0.0, last_err


# ---------------------------------------------------------------- OpenReview 兜底

VENUE_RE = re.compile(r"neurips\s*2025", re.I)


def content_val(c, key):
    v = c.get(key)
    return v.get("value") if isinstance(v, dict) else v


def fetch_openreview(title, want_group):
    qt = norm_tokens(title)
    want = (want_group or "").lower().strip()
    last_err = "no notes"
    for q in search_query_variants(title):
        url = "https://api2.openreview.net/notes/search?term=%s&limit=12" % urllib.parse.quote(q)
        body, err = http_get_retry(url)
        if body is None:
            last_err = err or "openreview empty"
            continue
        try:
            notes = json.loads(body).get("notes", []) or []
        except Exception as e:
            last_err = "json parse: %s" % str(e)[:80]
            continue
        if not notes:
            last_err = "no notes (term=%r)" % q[:40]
            continue
        best = None
        for n in notes:
            c = n.get("content", {}) or {}
            ct = content_val(c, "title") or ""
            venue = content_val(c, "venue") or ""
            ab = content_val(c, "abstract") or ""
            if not ab:
                continue
            s = score(qt, ct)
            bonus = 0.0
            if VENUE_RE.search(venue):
                gv = "spotlight" if "spotlight" in venue.lower() else (
                    "oral" if "oral" in venue.lower() else "neurips2025")
                bonus = 0.30 if gv == want else 0.12
            total = s + bonus
            if best is None or total > best[0]:
                best = (total, s, ab, venue, ct)
        if best is None:
            last_err = "notes without abstract"
            continue
        total, s, ab, venue, ct = best
        if s < 0.60:
            last_err = "low score %.2f (best=%s)" % (s, ct[:70])
            continue
        return ab, venue, ct, None
    return "", "", "", last_err


# ---------------------------------------------------------------- 进度

def load_rows():
    rows = []
    with open(CSV_IN, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            title = (r.get("论文标题") or "").strip()
            if not title:
                continue
            rows.append({
                "group": (r.get("分组") or "").strip(),
                "tag1": (r.get("主题标签1") or "").strip(),
                "tag2": (r.get("主题标签2") or "").strip(),
                "title": title,
            })
    return rows


def load_done():
    done = {}
    if not os.path.exists(PROGRESS):
        return done
    with open(PROGRESS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            done[d.get("title")] = d
    return done


def append(rec):
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def process(row):
    title, group = row["title"], row["group"]
    ab, aid, aurl, mt, pub, sc, err = fetch_arxiv(title)
    src = "arxiv"
    venue = ""
    if not ab:
        time.sleep(OR_DELAY)
        ab2, venue, mt2, err2 = fetch_openreview(title, group)
        if ab2:
            ab, mt, src = ab2, mt2, "openreview"
        else:
            err = "arxiv: %s | openreview: %s" % (err, err2)
    rec = {
        "title": title, "group": group, "tag1": row["tag1"], "tag2": row["tag2"],
        "abstract": ab or "", "source": src if ab else "",
        "arxiv_id": aid or "", "arxiv_url": aurl or "",
        "matched_title": mt or "", "published": pub or "",
        "venue_official": venue or "", "score": sc,
        "status": "ok" if ab else "miss", "error": "" if ab else (err or ""),
        "chars": len(ab or ""), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return rec


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--retry-miss", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    rows = load_rows()
    done = load_done()
    ok = [t for t, d in done.items() if d.get("status") == "ok"]
    miss = [t for t, d in done.items() if d.get("status") != "ok"]
    n_arx = sum(1 for d in done.values() if d.get("source") == "arxiv")
    log("CSV 总数=%d | 已成功=%d（arXiv %d） | 已失败=%d | 未处理=%d"
        % (len(rows), len(ok), n_arx, len(miss), len(rows) - len(done)))
    if args.report:
        return

    todo = []
    for r in rows:
        d = done.get(r["title"])
        if d is None or (d.get("status") != "ok" and args.retry_miss):
            todo.append(r)
    if args.max:
        todo = todo[:args.max]
    if not todo:
        log("没有待处理条目。")
        return

    log("本次待处理=%d 条，预计约 %.1f 分钟" % (len(todo), len(todo) * (ARXIV_DELAY + 0.6) / 60.0))
    n_ok = n_miss = 0
    t0 = time.time()
    for i, row in enumerate(todo, 1):
        try:
            rec = process(row)
        except KeyboardInterrupt:
            log("中断，已完成 %d/%d，可重跑续传。" % (i - 1, len(todo)))
            return
        except Exception as e:
            rec = {"title": row["title"], "group": row["group"], "tag1": row["tag1"],
                   "tag2": row["tag2"], "abstract": "", "source": "", "arxiv_id": "",
                   "arxiv_url": "", "matched_title": "", "published": "", "venue_official": "",
                   "score": 0.0, "status": "error",
                   "error": "%s: %s" % (type(e).__name__, str(e)[:160]), "chars": 0,
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        append(rec)
        n_ok += rec["status"] == "ok"
        n_miss += rec["status"] != "ok"
        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            log("进度 %d/%d | 成功 %d | 失败 %d | 命中率 %.1f%% | 已用 %.1f 分 | 剩余约 %.1f 分"
                % (i, len(todo), n_ok, n_miss, 100.0 * n_ok / i, el / 60.0,
                   el / i * (len(todo) - i) / 60.0))
        time.sleep(ARXIV_DELAY)

    log("本轮完成：成功 %d，失败 %d" % (n_ok, n_miss))


if __name__ == "__main__":
    main()

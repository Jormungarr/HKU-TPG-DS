# -*- coding: utf-8 -*-
"""
抓取 NeurIPS 2025 Oral / Spotlight 论文的官方 abstract。

数据源：
  1) OpenReview API v2  (主，venue 字段可直接校验 oral/spotlight 分组)
  2) arXiv API          (兜底)

设计要点：
  - 标题归一化 + 候选打分，而非字符串精确匹配。
    源 CSV 的标题做过转义：":" / "?" -> "_"，LaTeX 被压平成 $_ell_p$ 之类。
  - 断点续跑：每篇处理完立即 append 到 progress.jsonl，重启自动跳过已完成项。
    这是"长时间运行不中断"的真正保障，与内存无关。
  - 限流 + 指数退避重试；单篇彻底失败只记录状态，不影响整体流程。

用法：
  python fetch_abstracts.py --max 12 --sample   # 抽样验证匹配逻辑
  python fetch_abstracts.py                     # 跑全部待处理项（可中断可续跑）
  python fetch_abstracts.py --report            # 只输出当前进度报告
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
PROGRESS = os.path.join(OUTDIR, "progress.jsonl")

UA = "Mozilla/5.0 (compatible; course-abstract-fetcher/1.0)"
OR_DELAY = 1.2      # OpenReview 请求间隔（秒）
ARXIV_DELAY = 3.0   # arXiv 官方要求间隔 >= 3s
TIMEOUT = 30
MAX_RETRY = 3

socket.setdefaulttimeout(TIMEOUT)

# ---------------------------------------------------------------- 标题归一化

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
    """把标题压成可比对的 token 列表，抹掉转义差异与 LaTeX 噪声。"""
    t = (title or "").lower()
    t = re.sub(r"\\([a-zA-Z]+)", lambda m: " " + LATEX_MAP.get(m.group(1), m.group(1)) + " ", t)
    t = t.replace("$", " ")
    t = re.sub(r"[_{}~^\\]", " ", t)
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    toks = [w for w in t.split() if len(w) > 1 and w not in STOP]
    return toks


def search_query_variants(title):
    """生成多个由精确到宽松的检索查询，用于逐级降级重试。

    源 CSV 标题带转义（":" "?" -> "_"）与 LaTeX 噪声，直接检索易落空，
    因此先还原标点为空格，再逐步缩短。
    """
    t = re.sub(r"\\([a-zA-Z]+)", lambda m: " " + LATEX_MAP.get(m.group(1), m.group(1)) + " ",
               (title or "").lower())
    t = t.replace("$", " ").replace("&", " and ")
    t = re.sub(r"[_{}~^\\]", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    toks = [w for w in t.split() if len(w) > 1 and w not in STOP]
    variants = []
    if toks:
        variants.append(" ".join(toks))                 # 全词（去停用词）
        variants.append(" ".join(toks[:12]))            # 前 12 词
        if len(toks) > 6:
            variants.append(" ".join(toks[:6]))         # 前 6 词（最宽松）
    seen, out = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def score(query_toks, cand_title):
    """containment 打分：候选标题覆盖了多少查询 token，再对长度差做轻微惩罚。"""
    ct = norm_tokens(cand_title)
    if not query_toks or not ct:
        return 0.0
    qs, cs = set(query_toks), set(ct)
    inter = len(qs & cs)
    cover = inter / len(qs)
    ratio = min(len(qs), len(cs)) / max(len(qs), len(cs))
    return round(cover * 0.85 + ratio * 0.15, 4)


# ---------------------------------------------------------------- HTTP

def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json, text/xml, */*"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", "replace")


def http_get_retry(url, label):
    last = None
    for i in range(MAX_RETRY):
        try:
            return http_get(url), None
        except urllib.error.HTTPError as e:
            last = "HTTP %s" % e.code
            if e.code in (403, 404, 410):
                return None, last           # 不可重试
            time.sleep(2.0 * (i + 1))
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:120])
            time.sleep(2.0 * (i + 1))
    return None, last


# ---------------------------------------------------------------- 数据源

VENUE_RE = re.compile(r"neurips\s*2025", re.I)


def content_val(c, key):
    v = c.get(key)
    if isinstance(v, dict):
        return v.get("value")
    return v


def fetch_openreview(title, want_group):
    """返回 (abstract, venue, matched_title, err)；查询逐级降级重试。"""
    qt = norm_tokens(title)
    want = (want_group or "").lower().strip()
    last_err = "no notes"
    for q in search_query_variants(title):
        url = ("https://api2.openreview.net/notes/search?term=%s&limit=12"
               % urllib.parse.quote(q))
        body, err = http_get_retry(url, "openreview")
        if body is None:
            last_err = err or "openreview empty"
            time.sleep(1.0)
            continue
        try:
            notes = json.loads(body).get("notes", []) or []
        except Exception as e:
            last_err = "json parse: %s" % str(e)[:80]
            continue
        if not notes:
            last_err = "no notes (term=%r)" % q[:40]
            time.sleep(0.6)
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
            vm = VENUE_RE.search(venue)
            gv = ""
            if vm:
                gv = "spotlight" if "spotlight" in venue.lower() else (
                     "oral" if "oral" in venue.lower() else "neurips2025")
            bonus = 0.0
            if gv == want:
                bonus = 0.30          # 分组与 CSV 一致，强力加分
            elif vm:
                bonus = 0.12          # 是 NeurIPS 2025 但分组标记不同
            total = s + bonus
            cand = (total, s, ab, venue, ct, gv)
            if best is None or total > best[0]:
                best = cand
        if best is None:
            last_err = "notes without abstract"
            continue
        total, s, ab, venue, ct, gv = best
        if s < 0.60:
            last_err = "low score %.2f (best=%s)" % (s, ct[:70])
            continue                  # 换更宽松的查询再试
        return ab, venue, ct, None
    return None, None, None, last_err


ARX_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
ARX_T = re.compile(r"<title>(.*?)</title>", re.S)
ARX_S = re.compile(r"<summary>(.*?)</summary>", re.S)


def unescape_xml(s):
    return (s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"').replace("&apos;", "'").replace("&#39;", "'"))


def fetch_arxiv(title):
    """兜底：arXiv 标题检索，同样逐级降级。返回 (abstract, matched_title, err)"""
    last_err = "no entries"
    qt = norm_tokens(title)
    for q in search_query_variants(title)[:2]:        # arXiv 较慢，只试前两级
        query = urllib.parse.quote('ti:"%s"' % q)
        url = "http://export.arxiv.org/api/query?search_query=%s&max_results=6" % query
        body, err = http_get_retry(url, "arxiv")
        if body is None:
            last_err = err or "arxiv empty"
            continue
        entries = ARX_ENTRY.findall(body)
        if not entries:
            last_err = "no entries (term=%r)" % q[:40]
            continue
        best = None
        for e in entries:
            mt = ARX_T.search(e)
            ms = ARX_S.search(e)
            if not mt or not ms:
                continue
            ct = unescape_xml(mt.group(1)).strip()
            ab = unescape_xml(ms.group(1)).strip()
            s = score(qt, ct)
            if best is None or s > best[0]:
                best = (s, ab, ct)
        if best is None:
            last_err = "entries without abstract"
            continue
        s, ab, ct = best
        if s < 0.60:
            last_err = "low score %.2f (best=%s)" % (s, ct[:70])
            continue
        return ab, ct, None
    return None, None, last_err


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
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)


# ---------------------------------------------------------------- 主流程

def process(row):
    title, group = row["title"], row["group"]
    ab, venue, mt, err = fetch_openreview(title, group)
    src = "openreview"
    if not ab:
        time.sleep(ARXIV_DELAY)
        ab2, mt2, err2 = fetch_arxiv(title)
        if ab2:
            ab, mt, src = ab2, mt2, "arxiv"
        else:
            err = "%s | arxiv: %s" % (err, err2)
    rec = {
        "title": title,
        "group": group,
        "tag1": row["tag1"],
        "tag2": row["tag2"],
        "abstract": ab or "",
        "source": src if ab else "",
        "venue_official": venue or "",
        "matched_title": mt or "",
        "group_match": "",
        "status": "ok" if ab else "miss",
        "error": "" if ab else (err or ""),
        "chars": len(ab or ""),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if ab and venue:
        vl = venue.lower()
        off = "spotlight" if "spotlight" in vl else ("oral" if "oral" in vl else "")
        rec["group_match"] = ("一致" if off == group.lower() else
                              ("不一致" if off else "无法判定"))
    return rec


def main():
    try:                                    # Windows 控制台下避免中文乱码
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="本次最多处理多少条待办（0=不限）")
    ap.add_argument("--sample", action="store_true", help="等距抽样，用于验证匹配逻辑")
    ap.add_argument("--retry-miss", action="store_true", help="重跑此前失败的条目")
    ap.add_argument("--report", action="store_true", help="只输出进度报告")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    rows = load_rows()
    done = load_done()

    ok = [t for t, d in done.items() if d.get("status") == "ok"]
    miss = [t for t, d in done.items() if d.get("status") != "ok"]
    log("CSV 论文总数=%d | 已成功=%d | 已失败=%d | 未处理=%d"
        % (len(rows), len(ok), len(miss), len(rows) - len(done)))

    if args.report:
        return

    todo = []
    for r in rows:
        d = done.get(r["title"])
        if d is None:
            todo.append(r)
        elif d.get("status") != "ok" and args.retry_miss:
            todo.append(r)

    if args.sample and todo:
        n = min(args.max or 12, len(todo))
        step = max(1, len(todo) // n)
        todo = todo[::step][:n]
    elif args.max:
        todo = todo[:args.max]

    if not todo:
        log("没有待处理条目。")
        return

    log("本次待处理=%d 条，预计耗时约 %.1f 分钟"
        % (len(todo), len(todo) * (OR_DELAY + 0.5) / 60.0))

    n_ok = n_miss = 0
    t0 = time.time()
    for i, row in enumerate(todo, 1):
        try:
            rec = process(row)
        except KeyboardInterrupt:
            log("收到中断信号，已完成 %d/%d，可直接重跑续传。" % (i - 1, len(todo)))
            return
        except Exception as e:
            rec = {"title": row["title"], "group": row["group"], "tag1": row["tag1"],
                   "tag2": row["tag2"], "abstract": "", "source": "", "venue_official": "",
                   "matched_title": "", "group_match": "", "status": "error",
                   "error": "%s: %s" % (type(e).__name__, str(e)[:160]), "chars": 0,
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        append(rec)
        if rec["status"] == "ok":
            n_ok += 1
        else:
            n_miss += 1
        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            log("进度 %d/%d | 成功 %d | 失败 %d | 命中率 %.1f%% | 已用 %.1f 分钟 | 剩余约 %.1f 分钟"
                % (i, len(todo), n_ok, n_miss, 100.0 * n_ok / i, el / 60.0,
                   el / i * (len(todo) - i) / 60.0))
        time.sleep(OR_DELAY)

    log("本轮完成：成功 %d，失败 %d" % (n_ok, n_miss))
    if n_miss:
        log("失败样例（前 5 条）：")
        c = 0
        for t in todo:
            d = load_done().get(t["title"])
            if d and d.get("status") != "ok":
                log("  - [%s] %s -> %s" % (d.get("status"), t["title"][:60], d.get("error", "")[:110]))
                c += 1
                if c >= 5:
                    break


if __name__ == "__main__":
    main()

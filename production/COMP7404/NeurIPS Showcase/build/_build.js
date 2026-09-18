// 构建脚本：解析分类清单 CSV 与 oral.md 中文速读总结，生成自包含 HTML 检索页与站点入口页
const fs = require("fs");
const path = require("path");

const dir = __dirname;
const SITE_DIR = path.join(dir, "..");
const CSV_FILE = path.join(dir, "NeurIPS2025_oral_spotlight_论文分类清单.csv");
const TPL_FILE = path.join(dir, "_template.html");
const LANDING_TPL = path.join(dir, "_landing.html");
const OUT_FILE = path.join(SITE_DIR, "neurips2025-oral-spotlight", "index.html");
const LANDING_FILE = path.join(SITE_DIR, "index.html");
// oral.md：Oral 论文的中文速读总结，命中后替代该篇的原始摘要
const ORAL_MD = path.join(SITE_DIR, "oral.md");
// 摘要数据源（按顺序读取，后者覆盖前者）：OpenReview 历史结果 + arXiv 新结果
const ABSTRACT_FILES = [
  path.join(dir, "..", "abstracts", "progress.jsonl"),
  path.join(dir, "..", "abstracts", "arxiv_progress.jsonl"),
];

function parseCSV(text) {
  const rows = [];
  let row = [], field = "", inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQ = false;
      } else field += c;
    } else {
      if (c === '"') inQ = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
      else if (c !== "\r") field += c;
    }
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

// 连字还原：论文标题里的 ﬁ / ﬂ 等连字不会被 NFKD 分解，必须先手工展开，
// 否则 "Uniﬁed" 与 "Unified" 会被判为不同标题。
const LIGATURES = {
  "\uFB00": "ff", "\uFB01": "fi", "\uFB02": "fl",
  "\uFB03": "ffi", "\uFB04": "ffl", "\uFB05": "ft", "\uFB06": "st",
};
function unfold(s) {
  return String(s || "")
    .replace(/[\uFB00-\uFB06]/g, c => LIGATURES[c])
    .normalize("NFKD");
}

// 标题归一化：忽略大小写、标点、全角符号与连字差异，仅保留字母数字
function normTitle(s) {
  return unfold(s).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}
// 归一化后去掉空格，用于前缀比对（避免 "RL_ Scaling" 之类分隔差异造成错位）
function normKey(s) {
  return normTitle(s).replace(/ /g, "");
}
function tokensOf(s) {
  return new Set(normTitle(s).split(" ").filter(Boolean));
}

// 读取摘要 JSONL，按归一化标题建立索引；空摘要不覆盖已获得的非空摘要
function loadAbstracts() {
  const byTitle = new Map();
  for (const f of ABSTRACT_FILES) {
    if (!fs.existsSync(f)) continue;
    const lines = fs.readFileSync(f, "utf8").split(/\r?\n/);
    for (const ln of lines) {
      const t = ln.trim();
      if (!t) continue;
      let d; try { d = JSON.parse(t); } catch (e) { continue; }
      if (!d || !d.title) continue;
      const key = normTitle(d.title);
      if (!key) continue;
      const rec = { a: d.abstract || "", u: d.arxiv_url || "", id: d.arxiv_id || "", s: d.source || "" };
      const prev = byTitle.get(key);
      if (prev && prev.a && !rec.a) continue;
      byTitle.set(key, rec);
    }
  }
  return byTitle;
}

// 从已生成的页面里取回注入的 DATA 数组（构建产物即数据快照）
function extractData(html) {
  const m = html.match(/const DATA = ([\s\S]+?);\n\nconst state = \{/);
  if (!m) return null;
  try {
    const data = JSON.parse(m[1]);
    return Array.isArray(data) ? data : null;
  } catch (e) { return null; }
}

// abstracts/*.jsonl 是本地抓取产物、未随仓库分发；缺失时从已生成的页面里回收摘要，
// 保证只凭仓库内容也能重建页面（jsonl 若恢复，仍以 jsonl 为准）。
function recoverAbstracts() {
  const byTitle = new Map();
  if (!fs.existsSync(OUT_FILE)) return byTitle;
  const data = extractData(fs.readFileSync(OUT_FILE, "utf8"));
  if (!data) return byTitle;
  for (const d of data) {
    if (!d || !d.n || !d.a) continue;
    byTitle.set(normTitle(d.n), { a: d.a, u: d.u || "", id: d.id || "", s: d.s || "" });
  }
  return byTitle;
}

// ---------- oral.md 中文速读解析 ----------
// 每篇结构固定：概括 / 背景-痛点-方案-效果（四条要点）/ 方案解释 / 效果总结 / 未来方向
const FIELD_RE = /^\*\*(概括|背景-痛点-方案-效果|方案解释|效果总结|未来方向)\*\*[：:]?[ \t]*(.*)$/;
const BULLET_RE = /^-\s*(背景|痛点|方案|效果)[：:]\s*(.*)$/;
const BULLET_ORDER = ["背景", "痛点", "方案", "效果"];

function squeeze(lines) {
  return lines.map(s => s.trim()).filter(Boolean).join(" ").replace(/[ \t]+/g, " ").trim();
}

function parseOralMd(file) {
  const text = fs.readFileSync(file, "utf8").replace(/\uFEFF/, "");
  const blocks = text.split(/\r?\n(?=##\s*论文)/).filter(b => /^##\s*论文/.test(b));

  return blocks.map(block => {
    const tm = block.match(/^##\s*论文\s*([0-9]+)\s*[：:]\s*(.+?)\s*$/m);
    if (!tm) throw new Error("oral.md 条目缺少「## 论文N：标题」标题行：" + block.slice(0, 60));
    const no = tm[1];
    const title = tm[2];
    const where = "oral.md 论文" + no + "（" + title + "）";

    // 按字段起始行切段，段内其余行归入该字段
    const fields = new Map();
    let cur = null;
    for (const ln of block.slice(tm.index + tm[0].length).split(/\r?\n/)) {
      const fm = ln.match(FIELD_RE);
      if (fm) { cur = fm[1]; fields.set(cur, fm[2] ? [fm[2]] : []); continue; }
      if (cur) fields.get(cur).push(ln);
    }

    const missing = ["概括", "背景-痛点-方案-效果", "方案解释", "效果总结", "未来方向"]
      .filter(k => !fields.has(k));
    if (missing.length) throw new Error(where + " 缺少字段: " + missing.join(" / "));

    const bullets = new Map();
    let bullet = null;
    for (const ln of fields.get("背景-痛点-方案-效果")) {
      const bm = ln.match(BULLET_RE);
      if (bm) { bullet = bm[1]; if (bullets.has(bullet)) throw new Error(where + " 要点重复: " + bullet); bullets.set(bullet, [bm[2]]); continue; }
      if (bullet) bullets.get(bullet).push(ln);
    }
    const badBullets = [...bullets.keys()].filter(k => !BULLET_ORDER.includes(k));
    if (bullets.size !== BULLET_ORDER.length || badBullets.length) {
      throw new Error(where + " 的要点应为 " + BULLET_ORDER.join("/") + " 四条，实际: " + [...bullets.keys()].join("/"));
    }

    const sm = {
      sum: squeeze(fields.get("概括")),
      bg: squeeze(bullets.get("背景")),
      pain: squeeze(bullets.get("痛点")),
      sol: squeeze(bullets.get("方案")),
      eff: squeeze(bullets.get("效果")),
      exp: squeeze(fields.get("方案解释")),
      res: squeeze(fields.get("效果总结")),
      fut: squeeze(fields.get("未来方向")),
    };
    const empty = Object.keys(sm).filter(k => !sm[k]);
    if (empty.length) throw new Error(where + " 存在空字段: " + empty.join("/"));

    return { no: Number(no), title, sm };
  });
}

// 标题匹配：oral.md 标题常是清单标题的简称（省略副标题、RL 与 Reinforcement Learning
// 互换、PDF 连字未还原），因此按 精确 → 前缀 → 词集合包含 → 词相似度 逐级回退；
// 任一档命中多篇即视为歧义并报错，避免错配。
function matchSummary(entry, pool) {
  const title = normTitle(entry.title);
  const key = normKey(entry.title);
  const tests = [
    ["标题完全一致", p => normTitle(p.n) === title],
    ["标题互为前缀", p => normKey(p.n).startsWith(key) || key.startsWith(normKey(p.n))],
    ["标题词集合包含", p => {
      const a = p.toks, b = entry.toks;
      const small = a.size <= b.size ? a : b, big = a.size <= b.size ? b : a;
      return small.size > 0 && [...small].every(x => big.has(x));
    }],
    ["标题词相似度 ≥ 0.6", p => {
      const inter = [...p.toks].filter(x => entry.toks.has(x)).length;
      return inter / (p.toks.size + entry.toks.size - inter) >= 0.6;
    }],
  ];
  for (const [why, test] of tests) {
    const hit = pool.filter(p => !p.used && test(p));
    if (hit.length === 1) return { row: hit[0], why };
    if (hit.length > 1) {
      throw new Error("oral.md 标题「" + entry.title + "」在「" + why + "」下命中多篇: "
        + hit.map(p => p.n).join(" ／ "));
    }
  }
  return null;
}

const csv = fs.readFileSync(CSV_FILE, "utf8").replace(/^\uFEFF/, "");
const rows = parseCSV(csv);
const header = rows.shift();
if (header.join(",") !== "分组,主题标签1,主题标签2,论文标题") {
  throw new Error("CSV 表头不符: " + JSON.stringify(header));
}

const abs = loadAbstracts();
const recovered = recoverAbstracts();
let recUsed = 0;
for (const [k, v] of recovered) if (!abs.has(k)) { abs.set(k, v); recUsed++; }
let absHit = 0, urlHit = 0;
const data = rows
  .filter(r => r.length >= 4 && r[0].trim())
  .map(r => {
    const n = r[3].trim();
    const rec = abs.get(normTitle(n)) || {};
    if (rec.a) absHit++;
    if (rec.u) urlHit++;
    return { g: r[0].trim(), t: [r[1].trim(), r[2].trim()].filter(Boolean), n,
             a: rec.a || "", u: rec.u || "", id: rec.id || "", s: rec.s || "" };
  });

// 标签去重：CSV 中存在同一行两个标签相同的情况，会导致统计口径不一致
const dupRows = data.filter(d => new Set(d.t).size !== d.t.length).length;
if (dupRows) console.log("警告: " + dupRows + " 行存在重复标签，已自动去重");
data.forEach(d => { d.t = [...new Set(d.t)]; });

const bad = data.filter(d => !d.n || d.t.length === 0);
if (bad.length) throw new Error("存在缺少标题或标签的行: " + JSON.stringify(bad.slice(0, 3)));

const stats = { total: data.length };
stats.oral = data.filter(d => d.g === "oral").length;
stats.other = data.length - stats.oral;
if (stats.other !== data.filter(d => d.g === "spotlight").length) {
  throw new Error("存在异常的分组值: " + JSON.stringify([...new Set(data.map(d => d.g))]));
}

// ---------- 把中文速读挂到对应的 Oral 论文上 ----------
const entries = parseOralMd(ORAL_MD).map(e => ({ ...e, toks: tokensOf(e.title) }));
// 只在 Oral 分组内匹配：oral.md 只覆盖 Oral，限定范围可让错配直接暴露而不是静默落到 Spotlight
const pool = data.filter(d => d.g === "oral").map(d => ({ d, n: d.n, toks: tokensOf(d.n), used: false }));
const fuzzy = [];
for (const e of entries) {
  const hit = matchSummary(e, pool);
  if (!hit) throw new Error("oral.md 条目「" + e.title + "」在分类清单的 Oral 论文里找不到对应标题");
  hit.row.used = true;
  hit.row.d.sm = e.sm;
  if (hit.why !== "标题完全一致") fuzzy.push("  " + e.title + "  →  " + hit.row.n + "  [" + hit.why + "]");
}
const unmatched = pool.filter(p => !p.used);
if (unmatched.length) {
  throw new Error("有 Oral 论文没有对应的速读总结: " + unmatched.map(p => p.n).join(" ／ "));
}
stats.sm = entries.length;

const json = JSON.stringify(data).replace(/<\//g, "<\\/");
const tpl = fs.readFileSync(TPL_FILE, "utf8");
const TPL_MARK = "const DATA = [];";
if (!tpl.includes(TPL_MARK)) throw new Error("模板缺少占位符: " + TPL_MARK);
// 用函数式替换：JSON 中的 $'、$`、$&、$$ 若作为字符串替换参数会被当作特殊模式，
// 导致模板片段被注入到 DATA 中间、脚本解析失败且数据泄露到页面。
fs.writeFileSync(OUT_FILE, tpl.replace(TPL_MARK, () => "const DATA = " + json + ";"), "utf8");

// 自检：脚本主体只能出现一次，否则说明占位符替换引入了重复片段
const out = fs.readFileSync(OUT_FILE, "utf8");
const marker = "function renderList(";
const occ = out.split(marker).length - 1;
if (occ !== 1) throw new Error("脚本注入异常：'" + marker + "' 出现 " + occ + " 次（应为 1）");
if (out.includes(TPL_MARK)) throw new Error("占位符未被替换: " + TPL_MARK);
// 自检：回读注入的 DATA，确认速读总结篇数与预期一致且页面可解析
const emitted = extractData(out);
if (!emitted) throw new Error("产物无法回读 DATA（注入的 JSON 解析失败）");
const summOcc = emitted.filter(d => d.sm).length;
if (summOcc !== stats.sm) {
  throw new Error("产物中速读总结数 " + summOcc + " 与预期 " + stats.sm + " 不一致");
}

// ---------- 站点入口页 ----------
const builtOn = new Date().toISOString().slice(0, 10);
const landing = fs.readFileSync(LANDING_TPL, "utf8").replace(/\{\{(\w+)\}\}/g, (s, k) => {
  const v = { TOTAL: stats.total, ORAL: stats.oral, SPOT: stats.other, SM: stats.sm, BUILT: builtOn }[k];
  if (v === undefined) throw new Error("入口页模板存在未知占位符: " + s);
  return String(v);
});
if (/\{\{\w+\}\}/.test(landing)) throw new Error("入口页模板存在未替换占位符");
fs.writeFileSync(LANDING_FILE, landing, "utf8");

console.log("OK total=" + stats.total + " oral=" + stats.oral + " spotlight=" + stats.other + " 速读=" + stats.sm);
console.log("abstract 覆盖=" + absHit + "/" + stats.total + "  arXiv链接=" + urlHit
  + (recUsed ? "（其中 " + recUsed + " 条由已生成页面回收，abstracts/*.jsonl 缺失）" : ""));
if (fuzzy.length) console.log("速读标题非精确匹配（已按模糊规则挂载）:\n" + fuzzy.join("\n"));
console.log("output=" + OUT_FILE + " size=" + fs.statSync(OUT_FILE).size + " bytes");
console.log("output=" + LANDING_FILE + " size=" + fs.statSync(LANDING_FILE).size + " bytes");

// 构建脚本：解析分类清单 CSV，生成自包含 HTML 检索页面
const fs = require("fs");
const path = require("path");

const dir = __dirname;
const CSV_FILE = path.join(dir, "NeurIPS2025_oral_spotlight_论文分类清单.csv");
const TPL_FILE = path.join(dir, "_template.html");
const OUT_FILE = path.join(dir, "..", "neurips2025-oral-spotlight", "index.html");
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

// 标题归一化：忽略大小写、标点与下划线转义差异，仅保留字母数字，避免匹配遗漏
function normTitle(s) {
  return String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
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

const csv = fs.readFileSync(CSV_FILE, "utf8").replace(/^\uFEFF/, "");
const rows = parseCSV(csv);
const header = rows.shift();
if (header.join(",") !== "分组,主题标签1,主题标签2,论文标题") {
  throw new Error("CSV 表头不符: " + JSON.stringify(header));
}

const abs = loadAbstracts();
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

console.log("OK total=" + stats.total + " oral=" + stats.oral + " spotlight=" + stats.other);
console.log("abstract 覆盖=" + absHit + "/" + stats.total + "  arXiv链接=" + urlHit);
console.log("output=" + OUT_FILE + " size=" + fs.statSync(OUT_FILE).size + " bytes");

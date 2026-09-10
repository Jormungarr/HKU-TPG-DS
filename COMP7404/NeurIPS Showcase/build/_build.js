// 构建脚本：解析分类清单 CSV，生成自包含 HTML 检索页面
const fs = require("fs");
const path = require("path");

const dir = __dirname;
const CSV_FILE = path.join(dir, "NeurIPS2025_oral_spotlight_论文分类清单.csv");
const TPL_FILE = path.join(dir, "_template.html");
const OUT_FILE = path.join(dir, "..", "neurips2025-oral-spotlight", "index.html");

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

const csv = fs.readFileSync(CSV_FILE, "utf8").replace(/^\uFEFF/, "");
const rows = parseCSV(csv);
const header = rows.shift();
if (header.join(",") !== "分组,主题标签1,主题标签2,论文标题") {
  throw new Error("CSV 表头不符: " + JSON.stringify(header));
}

const data = rows
  .filter(r => r.length >= 4 && r[0].trim())
  .map(r => ({ g: r[0].trim(), t: [r[1].trim(), r[2].trim()].filter(Boolean), n: r[3].trim() }));

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
fs.writeFileSync(OUT_FILE, tpl.replace(TPL_MARK, "const DATA = " + json + ";"), "utf8");

console.log("OK total=" + stats.total + " oral=" + stats.oral + " spotlight=" + stats.other);
console.log("output=" + OUT_FILE + " size=" + fs.statSync(OUT_FILE).size + " bytes");

// sync.js — HKU Moodle 课程材料同步（枚举 → 增量比对 → 下载 → 落盘 → 日报）
//
// 运行（在已登录的 ego 浏览器会话里）：
//   ego-browser nodejs < ego/sync.js
// 只干跑不下载，先看会做什么：
//   touch DRY && ego-browser nodejs < ego/sync.js; rm DRY
// 只跑部分课程：
//   echo DASC7011 > ONLY && ego-browser nodejs < ego/sync.js; rm ONLY
//
// 注意：ego 不透传外层环境变量，开关用标记文件（见下），
// 不能用 MOODLE_DRY=1 这种写法。

const fs = await import("node:fs/promises");
const fsSync = await import("node:fs");
const nodePath = await import("node:path");
const crypto = await import("node:crypto");

const ROOT = "/Users/yukuanzou/HKU/HKU-TPG-DS/moodle-sync";
const MIRROR = nodePath.join(ROOT, "mirror");
const OUT = nodePath.join(ROOT, "out");
const STATE = nodePath.join(ROOT, "state");
const REPORTS = nodePath.join(ROOT, "reports");
const MANIFEST = nodePath.join(STATE, "manifest.json");
const EVENTS = nodePath.join(STATE, "events.jsonl");
const SPACE = 2;

// ego 不把外层环境变量透传给 node 进程，所以用标记文件而不是 MOODLE_DRY。
// 干跑：touch DRY && ego-browser nodejs < ego/sync.js
const DRY = fsSync.existsSync(nodePath.join(ROOT, "DRY"));
const ONLY = fsSync.existsSync(nodePath.join(ROOT, "ONLY"))
  ? (await fs.readFile(nodePath.join(ROOT, "ONLY"), "utf8"))
      .split(",").map((s) => s.trim().toUpperCase()).filter(Boolean)
  : null;

const COURSES = [
  { code: "COMP7404", id: 139691 },
  { code: "DASC7011", id: 139946 },
  { code: "DASC7606", id: 139952 },
  { code: "STAT8017", id: 144964 },
];

// ---------------------------------------------------------------- 工具

// Moodle 给重名文件自动加 " (1)" " (3)" 后缀。你本地的手工命名已经去过这些后缀，
// 照抄原名会在你的目录里造出成对的近重复文件，所以统一归一化。
function normName(name) {
  if (!name) return name;
  return name.replace(/\s*\(\d+\)(?=\.[^.]*$|$)/, "");
}

// 章节名要当目录名用，清掉 macOS 上非法的字符。
function safeDir(name) {
  const s = (name || "").replace(/[/\\:]/g, "-").replace(/\s+/g, " ").trim();
  return s || "(未命名单元)";
}

const nowISO = () => new Date().toISOString().replace(/\.\d+Z$/, "Z");
const today = () => new Date().toISOString().slice(0, 10);

async function sha256File(p) {
  const buf = await fs.readFile(p);
  return crypto.createHash("sha256").update(buf).digest("hex");
}

// ---------------------------------------------------------------- 页面侧采集

// 这个函数会被序列化后送进页面执行，因此必须自包含：
// 不能引用外部变量，只能用 document / fetch / DOMParser。
const collectCourse = async () => {
  const clean = (el) => {
    if (!el) return "";
    const c = el.cloneNode(true);
    c.querySelectorAll(".accesshide, .sr-only").forEach((n) => n.remove());
    return (c.textContent || "").trim().replace(/\s+/g, " ");
  };

  const sections = [...document.querySelectorAll("li.section")];
  const items = [];

  sections.forEach((sec) => {
    const sname = (sec.getAttribute("data-sectionname") || "").trim() || "(unnamed)";
    sec.querySelectorAll("li.activity").forEach((act) => {
      const modtype = (act.className.match(/modtype_([a-z]+)/) || [])[1] || "?";
      const link = act.querySelector("a.aalink") || act.querySelector("a[href*='/mod/']");
      const href = link ? link.href : null;
      const m = href && href.match(/[?&]id=(\d+)/);
      const nameEl = act.querySelector(".instancename");
      items.push({
        section: sname,
        modtype,
        modid: m ? m[1] : null,
        name: clean(nameEl),
        href,
        rawText: (act.textContent || "").trim().replace(/\s+/g, " ").slice(0, 200),
        indented: /\bindented\b/.test(act.className),
      });
    });
  });

  // 用 HEAD 跟一遍重定向拿到真实文件地址，不传输正文。
  const resolveFile = async (id) => {
    try {
      const r = await fetch("/mod/resource/view.php?id=" + id, {
        method: "HEAD",
        redirect: "follow",
      });
      const u = r.url;
      const cd = r.headers.get("content-disposition") || "";
      const fromCd = (cd.match(/filename="?([^";]+)"?/) || [])[1];
      const fromUrl = decodeURIComponent((u.split("/").pop() || "").split("?")[0]);
      return {
        moodleName: fromCd || fromUrl,
        mime: (r.headers.get("content-type") || "").split(";")[0],
        lastMod: r.headers.get("last-modified") || null,
        revision: (u.match(/\/content\/(\d+)\//) || [])[1] || null,
        fileUrl: u,
      };
    } catch (e) {
      return { error: String(e) };
    }
  };

  for (const it of items) {
    if (it.modtype === "resource" && it.modid) it.file = await resolveFile(it.modid);
  }

  // folder 模块要进到文件夹页里才能看到内部文件清单。
  for (const it of items.filter((i) => i.modtype === "folder" && i.modid)) {
    try {
      const r = await fetch("/mod/folder/view.php?id=" + it.modid);
      const doc = new DOMParser().parseFromString(await r.text(), "text/html");
      const seen = new Set();
      it.contents = [];
      doc.querySelectorAll("a[href*='/pluginfile.php/']").forEach((a) => {
        if (seen.has(a.href)) return;
        seen.add(a.href);
        const row = a.closest("tr") || a.closest("div");
        it.contents.push({
          moodleName: a.textContent.trim(),
          fileUrl: a.href,
          revision: (a.href.match(/\/content\/(\d+)\//) || [])[1] || null,
          rowText: row ? row.textContent.trim().replace(/\s+/g, " ").slice(0, 120) : null,
        });
      });
    } catch (e) {
      it.error = String(e);
    }
  }

  return {
    title: document.title,
    sections: sections.map((s) => (s.getAttribute("data-sectionname") || "").trim()),
    items,
  };
};

// ---------------------------------------------------------------- 主流程

const task = await takeOverTaskSpace(SPACE);
const page = task.page("p1");

await fs.mkdir(MIRROR, { recursive: true });
await fs.mkdir(OUT, { recursive: true });
await fs.mkdir(STATE, { recursive: true });
await fs.mkdir(REPORTS, { recursive: true });

let manifest = { version: 1, lastRun: null, courses: {}, files: {} };
let firstRun = true;
try {
  manifest = JSON.parse(await fs.readFile(MANIFEST, "utf8"));
  firstRun = false;
} catch (_) {}

const prevFiles = manifest.files || {};
const nextFiles = {};
const events = [];
const coursesWanted = COURSES.filter((c) => !ONLY || ONLY.includes(c.code));

const stats = { scanned: 0, newFiles: [], updated: [], unchanged: 0, errors: [],
                skippedDupe: [], nonFile: [] };

console.log(DRY ? "== 干跑模式：只比对，不下载 ==\n" : "== 同步模式 ==\n");

for (const course of coursesWanted) {
  await page.goto("https://moodle.hku.hk/course/view.php?id=" + course.id);
  await page.waitForLoadState("load", { timeout: 60000 }).catch(() => {});
  const data = await page.evaluate(collectCourse);
  stats.scanned++;

  const work = []; // 本次要处理的文件条目

  for (const it of data.items) {
    if (it.modtype === "resource" && it.file && it.file.fileUrl) {
      work.push({
        key: course.code + ":m" + it.modid,
        section: it.section,
        moduleName: it.name,
        moodleName: it.file.moodleName,
        fileUrl: it.file.fileUrl,
        mime: it.file.mime,
        lastMod: it.file.lastMod,
        revision: it.file.revision,
      });
    } else if (it.modtype === "folder" && it.contents) {
      for (const c of it.contents) {
        work.push({
          key: course.code + ":m" + it.modid + "/" + c.moodleName,
          section: it.section,
          moduleName: it.name,
          moodleName: c.moodleName,
          fileUrl: c.fileUrl,
          mime: null,
          lastMod: null,
          revision: c.revision,
        });
      }
    } else if (!["label", "subsection"].includes(it.modtype)) {
      // 非文件条目（page / url / assign / forum / feedback）也要留痕，
      // 否则它们会静默消失，你无从知道"还有东西没拿到"。
      stats.nonFile.push({
        course: course.code, section: it.section, modtype: it.modtype,
        name: it.name, href: it.href, text: it.rawText,
      });
    }
  }

  console.log("[" + course.code + "] " + data.sections.length + " 单元 / " +
              data.items.length + " 活动 / " + work.length + " 个文件");

  for (const w of work) {
    const name = normName(w.moodleName);
    const rel = nodePath.join(course.code, safeDir(w.section), name);
    const dest = nodePath.join(MIRROR, rel);
    const prev = prevFiles[w.key];

    const entry = {
      course: course.code,
      section: w.section,
      moduleName: w.moduleName,
      moodleName: w.moodleName,
      name,
      mime: w.mime,
      lastMod: w.lastMod,
      revision: w.revision,
      fileUrl: w.fileUrl,
      localPath: nodePath.relative(ROOT, dest),
      firstSeen: prev ? prev.firstSeen : nowISO(),
      lastSeen: nowISO(),
      sha256: prev ? prev.sha256 : null,
      size: prev ? prev.size : null,
      status: "unchanged",
    };

    const changed =
      !prev ||
      prev.revision !== w.revision ||
      prev.lastMod !== w.lastMod ||
      prev.moodleName !== w.moodleName;

    if (!prev) {
      entry.status = "new";
      stats.newFiles.push(entry);
    } else if (changed) {
      entry.status = "updated";
      stats.updated.push(entry);
    } else {
      stats.unchanged++;
    }

    if (entry.status !== "unchanged") {
      if (DRY) {
        console.log("   " + (entry.status === "new" ? "+ " : "~ ") + rel);
      } else {
        try {
          await fs.mkdir(nodePath.dirname(dest), { recursive: true });
          await page.fetch(w.fileUrl, { saveAs: dest });
          entry.size = (await fs.stat(dest)).size;
          entry.sha256 = await sha256File(dest);
          if (prev && prev.sha256 && prev.sha256 !== entry.sha256) {
            entry.prevSha256 = prev.sha256;
          }
          console.log("   " + (entry.status === "new" ? "+ " : "~ ") + rel +
                      "  " + entry.size + "B");
          events.push({ at: nowISO(), type: entry.status, course: course.code,
                        name, section: w.section, size: entry.size });
        } catch (e) {
          entry.status = "error";
          entry.error = String(e);
          stats.errors.push({ course: course.code, name, error: String(e) });
          console.log("   ! " + rel + "  FAILED: " + e);
        }
      }
    }
    nextFiles[w.key] = entry;
  }
}

// 上次有、这次没了 → 被老师删除或课程结构变动
const removed = Object.entries(prevFiles)
  .filter(([k, v]) => !nextFiles[k] && (!ONLY || ONLY.includes(v.course)))
  .map(([k, v]) => ({ key: k, ...v }));
for (const r of removed) {
  events.push({ at: nowISO(), type: "removed", course: r.course, name: r.name });
  console.log("   - " + r.course + "/" + r.section + "/" + r.name + "  (已从 Moodle 消失)");
}

// ---------------------------------------------------------------- 写状态

const summary = {
  at: nowISO(),
  dry: DRY,
  courses: coursesWanted.map((c) => c.code),
  totalFiles: Object.keys(nextFiles).length,
  new: stats.newFiles.length,
  updated: stats.updated.length,
  removed: removed.length,
  unchanged: stats.unchanged,
  errors: stats.errors.length,
};

if (!DRY) {
  for (const [k, v] of Object.entries(prevFiles)) if (!nextFiles[k]) delete nextFiles[k];
  manifest = {
    version: 1,
    lastRun: nowISO(),
    courses: Object.fromEntries(coursesWanted.map((c) => [c.code, c.id])),
    files: nextFiles,
  };
  await fs.writeFile(MANIFEST, JSON.stringify(manifest, null, 2));
  if (events.length) {
    await fs.appendFile(EVENTS, events.map((e) => JSON.stringify(e)).join("\n") + "\n");
  }
  await fs.writeFile(nodePath.join(OUT, "snapshot_" + today() + ".json"),
                     JSON.stringify({ summary, nonFile: stats.nonFile }, null, 2));
}

// ---------------------------------------------------------------- 报告

const byCourse = (list) => {
  const g = {};
  for (const x of list) (g[x.course] = g[x.course] || []).push(x);
  return g;
};

const L = [];
L.push("# Moodle 更新日报 · " + today());
L.push("");
L.push((DRY ? "> 干跑模式，未下载任何文件。" : "") +
       "扫描 " + summary.courses.join("、") + " 共 " +
       summary.totalFiles + " 个文件。");
L.push("");
L.push("| 新增 | 更新 | 消失 | 未变化 | 失败 |");
L.push("|---|---|---|---|---|");
L.push("| " + summary.new + " | " + summary.updated + " | " + summary.removed +
       " | " + summary.unchanged + " | " + summary.errors + " |");
L.push("");

if (stats.newFiles.length) {
  L.push("## 新增");
  L.push("");
  for (const [code, list] of Object.entries(byCourse(stats.newFiles))) {
    L.push("### " + code);
    L.push("");
    L.push("| 文件 | 单元 | 大小 | Moodle 原名 | 上传时间 |");
    L.push("|---|---|---|---|---|");
    for (const f of list) {
      L.push("| " + f.name + " | " + f.section + " | " +
             (f.size != null ? f.size + " B" : "—") + " | " +
             (f.moodleName !== f.name ? f.moodleName : "—") + " | " +
             (f.lastMod || "—") + " |");
    }
    L.push("");
  }
}

if (stats.updated.length) {
  L.push("## 更新（内容已变）");
  L.push("");
  for (const [code, list] of Object.entries(byCourse(stats.updated))) {
    L.push("### " + code);
    L.push("");
    L.push("| 文件 | 单元 | 修订号 | 上传时间 |");
    L.push("|---|---|---|---|");
    for (const f of list) {
      L.push("| " + f.name + " | " + f.section + " | " +
             (f.revision || "—") + " | " + (f.lastMod || "—") + " |");
    }
    L.push("");
  }
}

if (removed.length) {
  L.push("## 已从 Moodle 消失");
  L.push("");
  for (const r of removed) L.push("- " + r.course + " / " + r.section + " / " + r.name);
  L.push("");
}

if (stats.errors.length) {
  L.push("## 采集失败");
  L.push("");
  for (const e of stats.errors) L.push("- " + e.course + " / " + e.name + "：" + e.error);
  L.push("");
}

const nonFileSummary = {};
for (const n of stats.nonFile) {
  const k = n.course + " · " + n.modtype;
  nonFileSummary[k] = (nonFileSummary[k] || 0) + 1;
}
L.push("## 非文件条目（本轮未采集正文）");
L.push("");
L.push("这些不是文件，本轮只记录了它们的存在与位置，没有取正文：");
L.push("");
for (const [k, v] of Object.entries(nonFileSummary).sort()) L.push("- " + k + " × " + v);
L.push("");

const report = L.join("\n");
// 首次全量的报告单独命名，避免当天重跑（增量）把它覆盖掉。
const reportName = firstRun && !DRY
  ? today() + "_首次全量.md"
  : today() + ".md";
if (!DRY) {
  await fs.writeFile(nodePath.join(REPORTS, reportName), report);
  await fs.writeFile(nodePath.join(REPORTS, "latest.md"), report);
}
console.log("\n" + report);
console.log("报告：" + nodePath.join("reports", reportName));
console.log("清单：" + MANIFEST);

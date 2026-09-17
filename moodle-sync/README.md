# moodle-sync

HKU Moodle 课程材料同步：枚举 → 增量比对 → 下载 → 落盘 → 日报。

位于 `~/HKU/HKU-TPG-DS/` 下，与 `production`（个人产出仓库）并列：

```
~/HKU/HKU-TPG-DS/          学习材料收集（总目录）
├── production/            个人课程产出（公开 git 仓库）
└── moodle-sync/           课程材料镜像 + 同步工具（本目录）
```

分工：**别人给的**材料归 moodle-sync，**自己做的**东西归 production。

## 为什么是这个架构

港大 Moodle 用 CAS/SSO 认证，密码不存在 Moodle 里，所以官方 API 的
`login/token.php` 密码换取方式**永远返回 `invalidlogin`**；而 `moodle_mobile_app`
的 SSO 换 token 通道需要捕获 `moodlemobile://` 重定向，成本很高。同时
`lib/ajax/service.php` 在本站是关闭的（带真实会话也返回 `servicenotavailable`）。

结论：**唯一可用的凭据是浏览器里的活会话**。所以整个同步跑在 ego 浏览器里，
不碰 token、密码、cookie 的保存与传递。

## 前置条件

1. ego-browser 已安装（`~/.local/bin/ego-browser`）
2. 在一个 ego 任务空间里已登录 `moodle.hku.hk`
   （`task.handOff()` 让你本人完成登录，含 2FA）

## 用法

```bash
cd ~/HKU/HKU-TPG-DS/moodle-sync

# 正常同步
ego-browser nodejs < ego/sync.js

# 干跑：只比对，不下载
touch DRY && ego-browser nodejs < ego/sync.js; rm DRY

# 只跑部分课程
echo "DASC7011" > ONLY && ego-browser nodejs < ego/sync.js; rm ONLY
```

注意：ego **不透传外层环境变量**给 node 进程，所以开关用标记文件
（`DRY` / `ONLY`），不能用 `MOODLE_DRY=1` 这种写法。

## 目录

```
mirror/            课程材料，按 课程代码/单元名/ 落盘（不进 git）
mirror/_pages/     手工保存的 Moodle 页面导出（同步器尚未采集 page 正文，
                   暂时寄存在这里，共 3 个）
state/manifest.json  权威索引：修订号、修改时间、sha256、本地路径、Moodle 原名
state/events.jsonl   append-only 事件流，断点续跑的审计依据
reports/            日报；首次全量单独存为 YYYY-MM-DD_首次全量.md
                                    后续增量存 YYYY-MM-DD.md + latest.md
out/                运行日志与中间快照
```

## 设计要点

**变更检测不需要下载全文。** 每个 `mod_resource` 用一次 HEAD 请求跟重定向到
`pluginfile.php`，从 URL 里的修订号（`/content/N/`）和 `Last-Modified` 响应头
判断是否变化。folder 内的文件用 URL 里的修订号。只有真正新增或变化的才下载。

**文件名归一化。** Moodle 给重名文件自动加 ` (1)` ` (3)` 后缀，本地手工整理时
会去掉。磁盘上用归一化名（`DASC7011_T1 (1).pdf` → `DASC7011_T1.pdf`），
Moodle 原名完整保留在 manifest 和报告的"Moodle 原名"列，随时可追溯。

**镜像忠实于原件。** 下载内容逐字节等于 Moodle 原件，不做换行符转换。
（对照组：仓库里的 `DASC7011_Ch2.R` 是 3106 字节的 LF 版本，Moodle 原件是
3238 字节的 CRLF，已被 git 或编辑器改过。）

**非文件条目单独记录。** page / assign / forum / url / feedback 不是文件，
本轮只记录存在与位置，不取正文——避免它们静默消失、让你误以为已经拿全。

## 当前覆盖范围

- 课程：COMP7404、DASC7011、DASC7606、STAT8017（4 门学位课）
- 已采集：`mod_resource`（单个文件）、`mod_folder`（文件夹展开）
- 未采集正文：`mod_page`（网页正文）、`mod_assign`（截止日期与附件）、
  `mod_forum`（公告与附件）、`mod_url`（外链只记了标题）
- 未纳入：STAT_SAS / STAT_R / STAT_PYTHON / STAT_PROB / STAT_PRE_MC 五门预备课
- 文件大小：只有在下载时才知道（HEAD 返回的 `content-length` 不可靠）

## 已知约束

- 会话过期后需要重新在 ego 浏览器里登录一次，脚本才能继续
- 课程清单写死在 `ego/sync.js` 的 `COURSES` 里
- 首次全量、稳态增量的路径都已自检通过（无假阳性也无假阴性）

## 关于工具

只有 `ego/sync.js` 一个脚本。早期探索 token 路线时写的
`moodle_inventory.py` / `moodle_sso_token.py` 已删除——那条路线在港大
不可用（原因见上面"为什么是这个架构"），留死代码只会误导。

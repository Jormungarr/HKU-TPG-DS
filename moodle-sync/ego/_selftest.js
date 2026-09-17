// 自检：确认 ego 可以从脚本文件运行，并能读到已登录的会话。
const task = await takeOverTaskSpace(2);
const page = task.page("p1");
const info = await page.evaluate(() => ({
  url: location.href,
  loggedIn: !!(window.M && M.cfg && M.cfg.sesskey),
  userid: (window.M && M.cfg && M.cfg.userId) || null,
}));
console.log(JSON.stringify(info));

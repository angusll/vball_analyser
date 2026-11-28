import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const APP_PATH = path.resolve(import.meta.dirname, "..", "src", "vball_analyser", "static", "app.js");

function makeClassList(initial = []) {
  const set = new Set(initial);
  return {
    add: (...cls) => cls.forEach((c) => set.add(c)),
    remove: (...cls) => cls.forEach((c) => set.delete(c)),
    toggle: (c, force) => {
      const on = force === undefined ? !set.has(c) : Boolean(force);
      if (on) set.add(c);
      else set.delete(c);
      return on;
    },
    contains: (c) => set.has(c),
  };
}

function findAll(root, pred, out = []) {
  for (const c of root.children) {
    if (pred(c)) out.push(c);
    findAll(c, pred, out);
  }
  return out;
}

// Supports the selector forms app.js uses: [attr], [attr="v"], .class
function matcher(sel) {
  let m = /^\[([\w-]+)(?:="([^"]*)")?\]$/.exec(sel);
  if (m) return (n) => n.getAttribute(m[1]) !== null && (m[2] === undefined || n.getAttribute(m[1]) === m[2]);
  m = /^\.([\w-]+)$/.exec(sel);
  if (m) return (n) => String(n.className).split(/\s+/).includes(m[1]) || n.classList.contains(m[1]);
  throw new Error(`unsupported selector ${sel}`);
}

let doc = null;

function makeEl(tag, { id = "", classes = [], attrs = {} } = {}) {
  const listeners = new Map();
  const attributes = new Map(Object.entries(attrs));
  return {
    tagName: tag.toUpperCase(),
    id,
    classList: makeClassList(classes),
    children: [],
    parentNode: null,
    parentElement: null,
    style: {},
    dataset: {},
    className: "",
    textContent: "",
    value: "",
    title: "",
    disabled: false,
    setAttribute(k, v) { attributes.set(k, String(v)); },
    getAttribute(k) { return attributes.has(k) ? attributes.get(k) : null; },
    removeAttribute(k) { attributes.delete(k); },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(fn);
    },
    async dispatch(type, ev = {}) {
      const full = { preventDefault() {}, stopPropagation() {}, target: this, ...ev };
      for (const fn of listeners.get(type) || []) await fn(full);
    },
    appendChild(child) {
      child.parentNode = this;
      child.parentElement = this;
      this.children.push(child);
      return child;
    },
    replaceChildren(...kids) {
      this.children = [];
      kids.forEach((k) => this.appendChild(k));
    },
    querySelectorAll(sel) { return findAll(this, matcher(sel)); },
    querySelector(sel) { return findAll(this, matcher(sel))[0] || null; },
    focus() { doc.activeElement = this; },
  };
}

function allText(root) {
  return [root.textContent, ...findAll(root, () => true).map((n) => n.textContent)].join(" | ");
}

const flush = async () => {
  for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r));
};

function zone(name, count, pct, tempo, tempoStd, peak, peakStd, pin, pinStd) {
  return {
    zone: name, count, pct,
    tempo_mean_s: tempo, tempo_std_s: tempoStd,
    peak_height_mean_cm: peak, peak_height_std_cm: peakStd,
    pin_distance_mean_m: pin, pin_distance_std_m: pinStd,
  };
}

function blockStats(over = {}) {
  return {
    opposed_count: 0, unopposed_count: 1, zero_block_count: 0, single_block_count: 0,
    open_seam_count: 0, closed_block_count: 0, isolation_pct: null, closed_block_pct: null,
    avg_open_seam_cm: null, ...over,
  };
}

function bucket(prefix, over = {}) {
  return {
    play_count: 1, back_row_count: 0, back_row_pct: 0,
    zones: [
      zone(`${prefix}Zone 4 (Left)`, 2, 66.7, 0.52, 0.04, 61, 5, 0.9, 0.1),
      zone(`${prefix}Zone 3 (Middle)`, 1, 33.3, 0.31, null, 40, null, null, null),
      zone(`${prefix}Zone 2 (Right)`, 0, 0, null, null, null, null, null, null),
    ],
    block_stats: blockStats(),
    ...over,
  };
}

function makePlay(id, index, teamSide, attackZone) {
  return {
    play_id: id, index, start_time_s: index * 5, end_time_s: index * 5 + 3, start_frame: index * 150,
    attack_zone: attackZone, team_side: teamSide, team_side_manual: false,
    system_state: teamSide === "ours" ? "in_system" : null,
    pin_distance_m: teamSide === "ours" ? 0.9 : null,
    hitter_track_id: 1,
    set_trajectory: null,
    hitter_metrics: {
      contact_height_px: 20, contact_height_cm: 30, arm_extension_deg: 160, torso_tilt_deg: 12,
      jump_timing_label: "At Peak", is_back_row_approximate: false, dominant_hand: "right", contact_point: null,
    },
    block_summary: { blockers: [], seam_open: false, seam_gap_cm: null, seam_interval_x: null, block_state: "unopposed" },
    keyframes: [],
    tracks: [
      { track_id: 1, court_side: "attacking", frames: [{ frame_idx: 10, bbox: [100, 100, 140, 200], keypoints: [] }] },
      { track_id: 2, court_side: "blocking", frames: [{ frame_idx: 10, bbox: [105, 100, 145, 200], keypoints: [] }] },
    ],
  };
}

function makeBackend() {
  const session = {
    session_id: "s1",
    filename: "match.mp4",
    status: "ready",
    progress_pct: 100,
    session_mode: "drill",
    metadata: { width: 1280, height: 720, fps: 30 },
    calibration: { left_point: { x: 100, y: 200 }, right_point: { x: 1100, y: 200 }, net_height_m: 2.43, is_manual: true },
    plays: [makePlay("p1", 1, "ours", "Zone 4 (Left)"), makePlay("p2", 2, "opponent", "Opp Zone 2 (Right)")],
    summary: {
      session_mode: "drill",
      total_plays: 2, ours_count: 1, opponent_count: 1,
      in_system_count: 1, in_system_pct: 100,
      ours_all: bucket(""),
      ours_in_system: bucket(""),
      ours_out_of_system: bucket("", { play_count: 0 }),
      opponent: bucket("Opp ", {
        block_stats: blockStats({ opposed_count: 1, closed_block_count: 1, closed_block_pct: 100, avg_open_seam_cm: 27 }),
      }),
    },
  };
  const posts = [];
  const snapshot = () => JSON.parse(JSON.stringify(session));
  const fetchFake = async (url, opts = {}) => {
    if (url === "/api/csrf") return { ok: true, status: 200, json: async () => ({ csrf_token: "tok" }) };
    if (url === "/api/sessions") return { ok: true, status: 200, json: async () => [snapshot()] };
    if (opts.method === "POST") {
      const body = JSON.parse(opts.body);
      posts.push({ url, body });
      if (url.endsWith("/mode")) {
        session.session_mode = body.session_mode;
        session.summary.session_mode = body.session_mode;
      }
      const m = /\/plays\/([^/]+)\/side$/.exec(url);
      if (m) session.plays.find((p) => p.play_id === m[1]).team_side = body.team_side;
    }
    return { ok: true, status: 200, json: async () => snapshot() };
  };
  return { session, posts, fetchFake };
}

async function makeHarness(mutate = () => {}) {
  const backend = makeBackend();
  mutate(backend.session);
  const byId = new Map();
  const getEl = (id) => {
    if (!byId.has(id)) byId.set(id, makeEl("div", { id }));
    return byId.get(id);
  };
  const group = (id, attr, values) => {
    for (const v of values) getEl(id).appendChild(makeEl("button", { attrs: { [attr]: v } }));
  };
  group("studioModeGroup", "data-session-mode", ["drill", "match"]);
  group("playTeamFilterGroup", "data-team-filter", ["all", "ours", "opponent"]);
  group("studioSidebarViewGroup", "data-sidebar-view", ["play", "session"]);
  getEl("playTeamFilterGroup").classList.add("hidden");
  getEl("sessionSummaryContainer").classList.add("hidden");
  getEl("loadSessionModeSelect").value = "drill";

  doc = {
    activeElement: null,
    getElementById: getEl,
    createElement: (tag) => makeEl(tag),
    querySelectorAll: () => [],
  };
  const mp = {
    mountTo() {}, requestRender() {}, hasVideo: () => false, loadSession() {}, setCanvasInteractive() {},
    getSessionId: () => "s1", pause() {}, getCurrentFrame: () => 0, jumpToFrame() {}, getCurrentTime: () => 0,
  };
  const windowFake = { VBallMediaPlayer: { MediaPlayer: function MediaPlayer() { return mp; } } };
  const sandbox = {
    window: windowFake, document: doc, fetch: backend.fetchFake, console,
    setTimeout: () => 1, clearTimeout() {},
  };
  vm.runInNewContext(fs.readFileSync(APP_PATH, "utf-8"), sandbox, { filename: "app.js" });
  await flush();
  getEl("sessionSelect").value = "s1";
  await getEl("sessionSelect").dispatch("change");
  await flush();

  const btn = (groupId, attr, v) => getEl(groupId).children.find((b) => b.getAttribute(attr) === v);
  const playItems = () => getEl("playsList").children.filter((c) => c.className.includes("play-item"));
  return { backend, getEl, btn, playItems, sandbox };
}

test("Clicking Match POSTs session_mode and reveals the Ours | Opp filter", async () => {
  const h = await makeHarness((s) => { s.plays[1].team_side = "ours"; });
  assert.ok(h.getEl("playTeamFilterGroup").classList.contains("hidden"), "drill with only ours plays hides filter");
  assert.ok(h.btn("studioModeGroup", "data-session-mode", "drill").classList.contains("active"));

  await h.btn("studioModeGroup", "data-session-mode", "match").dispatch("click");
  await flush();
  const post = h.backend.posts.at(-1);
  assert.equal(post.url, "/api/sessions/s1/mode");
  assert.deepEqual(post.body, { session_mode: "match" });
  assert.ok(!h.getEl("playTeamFilterGroup").classList.contains("hidden"));
  assert.ok(h.btn("studioModeGroup", "data-session-mode", "match").classList.contains("active"));
  assert.equal(h.btn("studioModeGroup", "data-session-mode", "match").getAttribute("aria-pressed"), "true");
});

test("Opponent filter hides ours cards and moves the active play to a visible one", async () => {
  const h = await makeHarness();
  assert.ok(!h.getEl("playTeamFilterGroup").classList.contains("hidden"), "opponent plays reveal filter in drill");
  assert.equal(h.playItems().length, 2);
  assert.ok(allText(h.playItems()[0]).includes("Ours"));
  assert.ok(allText(h.playItems()[1]).includes("Opp"));

  await h.btn("playTeamFilterGroup", "data-team-filter", "opponent").dispatch("click");
  await flush();
  const items = h.playItems();
  assert.equal(items.length, 1);
  assert.ok(allText(items[0]).includes("Play #2"));
  assert.ok(items[0].className.includes("active"));
  assert.ok(h.btn("playTeamFilterGroup", "data-team-filter", "opponent").classList.contains("active"));
  const metrics = allText(h.getEl("metricsList"));
  assert.ok(metrics.includes("Our Block"), "opponent play labels block row Our Block");
  assert.ok(metrics.includes("Unopposed"));
  assert.ok(metrics.includes("Far-court est."));
  assert.ok(metrics.includes("Opponent"));
});

test("Swap Side POSTs team_side to the play side endpoint", async () => {
  const h = await makeHarness();
  await h.getEl("swapPlaySideBtn").dispatch("click");
  await flush();
  let post = h.backend.posts.at(-1);
  assert.equal(post.url, "/api/sessions/s1/plays/p1/side");
  assert.deepEqual(post.body, { team_side: "opponent" });

  await h.getEl("swapPlaySideBtn").dispatch("click");
  await flush();
  post = h.backend.posts.at(-1);
  assert.deepEqual(post.body, { team_side: "ours" });
  assert.ok(allText(h.getEl("metricsList")).includes("0.90 m from antenna"));
});

test("Session view renders set window and isolation, and match sub-views toggle cleanly", async () => {
  const h = await makeHarness();
  await h.btn("studioSidebarViewGroup", "data-sidebar-view", "session").dispatch("click");
  await flush();
  const container = h.getEl("sessionSummaryContainer");
  assert.ok(h.getEl("metricsList").classList.contains("hidden"));
  assert.ok(!container.classList.contains("hidden"));
  let txt = allText(container);
  for (const s of ["Z4", "Z3", "Z2", "0.52 ± 0.04 s", "0.31 s", "+61 ± 5 cm", "0.90 ± 0.10 m", "Setter Isolation", "Unopposed", "Our Reps"]) {
    assert.ok(txt.includes(s), `drill summary missing ${s}`);
  }
  assert.ok(!txt.includes("Our Offense"));

  await h.btn("studioModeGroup", "data-session-mode", "match").dispatch("click");
  await flush();
  txt = allText(container);
  assert.ok(txt.includes("Our Offense") && txt.includes("In-System") && txt.includes("Out-of-Sys"));
  assert.ok(txt.includes("In-System %"));

  const pill = (attr, v) => container.querySelector(`[${attr}="${v}"]`);
  await pill("data-summary-tab", "opponent").dispatch("click");
  txt = allText(container);
  assert.ok(txt.includes("Opp Z2") && txt.includes("Our Block Closure") && txt.includes("100%") && txt.includes("27 cm"));
  assert.ok(!txt.includes("Out-of-Sys"));
  assert.equal(h.sandbox.document.activeElement, pill("data-summary-tab", "opponent"), "focus restored to re-rendered pill");

  await pill("data-summary-tab", "ours").dispatch("click");
  await pill("data-ours-split", "out_of_system").dispatch("click");
  assert.ok(pill("data-ours-split", "out_of_system").classList.contains("active"));
  assert.equal(h.sandbox.document.activeElement, pill("data-ours-split", "out_of_system"));

  await h.btn("studioSidebarViewGroup", "data-sidebar-view", "play").dispatch("click");
  assert.ok(container.classList.contains("hidden"));
  assert.ok(!h.getEl("metricsList").classList.contains("hidden"));
});

test("findNearestAttackingTrackId targets the offense side of the play", async () => {
  const h = await makeHarness();
  const play = h.backend.session.plays[0];
  const pt = { x: 122, y: 150 };
  assert.equal(h.sandbox.findNearestAttackingTrackId({ ...play, team_side: "ours" }, pt, 10), 1);
  assert.equal(h.sandbox.findNearestAttackingTrackId({ ...play, team_side: "opponent" }, pt, 10), 2);
});

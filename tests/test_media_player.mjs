import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const MEDIA_PLAYER_PATH = path.resolve(
  import.meta.dirname,
  "..",
  "src",
  "vball_analyser",
  "static",
  "media_player.js"
);

function makeClassList(initial = []) {
  const set = new Set(initial);
  return {
    add: (...cls) => cls.forEach((c) => set.add(c)),
    remove: (...cls) => cls.forEach((c) => set.delete(c)),
    toggle: (c, force) => {
      if (force === undefined) {
        if (set.has(c)) { set.delete(c); return false; }
        set.add(c); return true;
      }
      if (force) set.add(c);
      else set.delete(c);
      return Boolean(force);
    },
    contains: (c) => set.has(c),
  };
}

function makeFakeElement(tag, { id = "", classes = [], attrs = {} } = {}) {
  const listeners = new Map();
  const attributes = new Map(Object.entries(attrs));
  const el = {
    tagName: tag.toUpperCase(),
    id,
    classList: makeClassList(classes),
    children: [],
    parentNode: null,
    disabled: false,
    textContent: "",
    value: "0",
    max: "1",
    clientWidth: 800,
    clientHeight: 450,
    width: 0,
    height: 0,
    setAttribute(k, v) { attributes.set(k, String(v)); },
    getAttribute(k) { return attributes.has(k) ? attributes.get(k) : null; },
    hasAttribute(k) { return attributes.has(k); },
    removeAttribute(k) { attributes.delete(k); },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(fn);
    },
    dispatch(type, ev = {}) {
      for (const fn of listeners.get(type) || []) fn(ev);
    },
    appendChild(child) {
      if (child.parentNode) {
        child.parentNode.children = child.parentNode.children.filter((c) => c !== child);
      }
      child.parentNode = this;
      this.children.push(child);
      return child;
    },
    getBoundingClientRect() {
      return { left: 100, top: 50, width: this.clientWidth, height: this.clientHeight };
    },
    getContext() {
      return {
        setTransform() {},
        clearRect() {},
        fillRect() {},
        beginPath() {},
        roundRect() {},
        fill() {},
        stroke() {},
        moveTo() {},
        lineTo() {},
        arc() {},
        save() {},
        restore() {},
        fillStyle: "",
        strokeStyle: "",
        lineWidth: 1,
      };
    },
  };
  return el;
}

function makeHarness() {
  const docListeners = new Map();
  const winListeners = new Map();

  const video = makeFakeElement("video", { id: "playerVideo" });
  video.videoWidth = 1280;
  video.videoHeight = 720;
  video.currentTime = 0;
  video.playbackRate = 1;
  video.paused = true;
  video.ended = false;
  video.duration = 20;
  video.src = "";
  video.play = () => {
    video.paused = false;
    video.dispatch("play");
  };
  video.pause = () => {
    video.paused = true;
    video.dispatch("pause");
  };
  video.load = () => {};

  const canvas = makeFakeElement("canvas", { id: "overlayCanvas" });
  const timelineCanvas = makeFakeElement("canvas", { id: "timelineCanvas" });
  timelineCanvas.clientWidth = 400;
  timelineCanvas.clientHeight = 24;
  const seekSlider = makeFakeElement("input", { id: "seekSlider" });
  const timeReadout = makeFakeElement("span", { id: "timeReadout" });
  const stepBack5Btn = makeFakeElement("button", { id: "stepBack5Btn" });
  const stepBack1Btn = makeFakeElement("button", { id: "stepBack1Btn" });
  const playPauseBtn = makeFakeElement("button", { id: "playPauseBtn" });
  const playPauseLabel = makeFakeElement("span", { id: "playPauseLabel" });
  const stepFwd1Btn = makeFakeElement("button", { id: "stepFwd1Btn" });
  const stepFwd5Btn = makeFakeElement("button", { id: "stepFwd5Btn" });
  const emptyState = makeFakeElement("div", { id: "emptyState" });
  const stageDropOverlay = makeFakeElement("div", { id: "stageDropOverlay", classes: ["hidden"] });
  const videoStage = makeFakeElement("div", { id: "videoStage" });

  const speed025 = makeFakeElement("button", { classes: ["speed-btn"], attrs: { "data-speed": "0.25", "aria-pressed": "false" } });
  const speed05 = makeFakeElement("button", { classes: ["speed-btn"], attrs: { "data-speed": "0.5", "aria-pressed": "false" } });
  const speed1 = makeFakeElement("button", { classes: ["speed-btn", "active"], attrs: { "data-speed": "1", "aria-pressed": "true" } });
  const externalFormationBtn = makeFakeElement("button", {
    classes: ["speed-btn", "serve-formation-btn", "active"],
    attrs: { "data-formation": "3" },
  });

  const bySelector = {
    "#playerVideo": video,
    "#overlayCanvas": canvas,
    "#timelineCanvas": timelineCanvas,
    "#seekSlider": seekSlider,
    "#timeReadout": timeReadout,
    "#stepBack5Btn": stepBack5Btn,
    "#stepBack1Btn": stepBack1Btn,
    "#playPauseBtn": playPauseBtn,
    "#playPauseLabel": playPauseLabel,
    "#stepFwd1Btn": stepFwd1Btn,
    "#stepFwd5Btn": stepFwd5Btn,
    "#emptyState": emptyState,
    "#stageDropOverlay": stageDropOverlay,
    "#videoStage": videoStage,
  };

  const root = makeFakeElement("div", { id: "mediaPlayerRoot", classes: ["media-player"] });
  root.querySelector = (sel) => bySelector[sel] || null;
  root.querySelectorAll = (sel) => (sel === ".speed-btn" ? [speed025, speed05, speed1] : []);

  const studioSlot = makeFakeElement("div", { id: "studioPlayerSlot" });
  const serveSlot = makeFakeElement("div", { id: "servePlayerSlot" });
  studioSlot.appendChild(root);

  const documentFake = {
    addEventListener(type, fn) {
      if (!docListeners.has(type)) docListeners.set(type, []);
      docListeners.get(type).push(fn);
    },
    dispatch(type, ev) {
      for (const fn of docListeners.get(type) || []) fn(ev);
    },
    querySelectorAll(sel) {
      if (sel === ".speed-btn") return [speed025, speed05, speed1, externalFormationBtn];
      return [];
    },
  };

  const windowFake = {
    devicePixelRatio: 1,
    addEventListener(type, fn) {
      if (!winListeners.has(type)) winListeners.set(type, []);
      winListeners.get(type).push(fn);
    },
    requestAnimationFrame: () => 1,
  };
  windowFake.window = windowFake;

  const sandbox = {
    window: windowFake,
    document: documentFake,
    requestAnimationFrame: windowFake.requestAnimationFrame,
    Math,
    Number,
    String,
    Boolean,
    Array,
    Object,
    encodeURIComponent,
    parseFloat,
    parseInt,
    isFinite,
  };
  sandbox.globalThis = sandbox;

  const code = fs.readFileSync(MEDIA_PLAYER_PATH, "utf-8");
  vm.runInNewContext(code, sandbox, { filename: "media_player.js" });

  const player = new windowFake.VBallMediaPlayer.MediaPlayer(root);
  return {
    player,
    root,
    video,
    canvas,
    seekSlider,
    timeReadout,
    emptyState,
    speed025,
    speed05,
    speed1,
    externalFormationBtn,
    studioSlot,
    serveSlot,
    documentFake,
  };
}

test("loadSession preserves currentTime and playbackRate when session_id is unchanged", () => {
  const h = makeHarness();
  const sessA = { session_id: "sess-1", metadata: { fps: 30, duration_s: 20, width: 1280, height: 720 } };
  const sessB = { session_id: "sess-2", metadata: { fps: 60, duration_s: 10, width: 1280, height: 720 } };

  assert.equal(h.player.loadSession(sessA, { resetTime: true }), true);
  assert.equal(h.video.src, "/api/sessions/sess-1/video");

  h.player.seekToTime(7.5);
  h.player.setPlaybackRate(0.5);
  assert.equal(h.player.loadSession(sessA, { resetTime: false }), false);
  assert.equal(h.player.getCurrentTime(), 7.5);
  assert.equal(h.video.playbackRate, 0.5);

  assert.equal(h.player.loadSession(sessB, { resetTime: true }), true);
  assert.equal(h.video.src, "/api/sessions/sess-2/video");
  assert.equal(h.player.getFps(), 60);
});

test("mountTo pauses playback, reparents root, swaps adapter, and renders", () => {
  const h = makeHarness();
  h.player.loadSession({ session_id: "sess-1", metadata: { fps: 30, duration_s: 20, width: 1280, height: 720 } });
  h.player.play();
  assert.equal(h.video.paused, false);

  let overlayCalls = 0;
  let timelineCalls = 0;
  h.player.mountTo(h.serveSlot, {
    showEmptyState: false,
    onRenderOverlay: () => { overlayCalls += 1; },
    onRenderTimeline: () => { timelineCalls += 1; },
  });

  assert.equal(h.video.paused, true);
  assert.equal(h.root.parentNode, h.serveSlot);
  assert.equal(h.emptyState.classList.contains("hidden"), true);
  assert.ok(overlayCalls >= 1);
  assert.ok(timelineCalls >= 1);
});

test("speed buttons are scoped to MediaPlayer root and do not touch external formation buttons", () => {
  const h = makeHarness();
  h.speed05.dispatch("click");
  assert.equal(h.video.playbackRate, 0.5);
  assert.equal(h.speed05.classList.contains("active"), true);
  assert.equal(h.speed1.classList.contains("active"), false);
  assert.equal(h.externalFormationBtn.classList.contains("active"), true);
});

test("stepFrames and jumpToFrame clamp within [0, duration]", () => {
  const h = makeHarness();
  h.player.loadSession({ session_id: "sess-1", metadata: { fps: 25, duration_s: 10, width: 1280, height: 720 } });

  h.player.jumpToFrame(50);
  assert.equal(h.player.getCurrentTime(), 2);
  assert.equal(h.player.getCurrentFrame(), 50);

  h.player.stepFrames(-100);
  assert.equal(h.player.getCurrentTime(), 0);

  h.player.stepFrames(500);
  assert.equal(h.player.getCurrentTime(), 10);
});

test("toCanvasPt and toVideoPt round-trip accurately", () => {
  const h = makeHarness();
  h.player.loadSession({ session_id: "sess-1", metadata: { fps: 30, duration_s: 10, width: 1280, height: 720 } });

  const orig = { x: 640, y: 360 };
  const canvasPt = h.player.toCanvasPt(orig);
  // getBoundingClientRect() has left=100, top=50
  const roundTrip = h.player.toVideoPt(100 + canvasPt.x, 50 + canvasPt.y);
  assert.ok(Math.abs(roundTrip.x - orig.x) < 1e-6);
  assert.ok(Math.abs(roundTrip.y - orig.y) < 1e-6);
});

test("keyboard shortcuts work in both tabs and ignore form inputs", () => {
  const h = makeHarness();
  h.player.loadSession({ session_id: "sess-1", metadata: { fps: 30, duration_s: 10, width: 1280, height: 720 } });
  let customKeys = 0;
  h.player.mountTo(h.serveSlot, {
    onKeyDown: () => { customKeys += 1; },
  });

  // Ignored inside INPUT
  let prevented = false;
  h.documentFake.dispatch("keydown", {
    code: "Space",
    target: { tagName: "INPUT" },
    preventDefault: () => { prevented = true; },
  });
  assert.equal(prevented, false);
  assert.equal(h.video.paused, true);

  // Space toggles playback on non-input target
  h.documentFake.dispatch("keydown", {
    code: "Space",
    target: { tagName: "BODY" },
    preventDefault: () => { prevented = true; },
  });
  assert.equal(prevented, true);
  assert.equal(h.video.paused, false);

  // ArrowRight steps +1 frame and pauses
  h.documentFake.dispatch("keydown", {
    code: "ArrowRight",
    target: { tagName: "BODY" },
    preventDefault: () => {},
  });
  assert.equal(h.video.paused, true);
  assert.equal(h.player.getCurrentFrame(), 1);
  assert.ok(customKeys >= 2);
});

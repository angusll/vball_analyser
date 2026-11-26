"use strict";

const COCO_EDGES = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];

const ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi"];

const state = {
  csrfToken: "",
  session: null,
  selectedPlayId: null,
  overlays: {
    setArc: true,
    pose: true,
    contact: true,
    blockers: true,
    netTape: true,
    hud: true,
    inlineStats: true,
  },
  calibratingNet: false,
  calFirstPoint: null,
  calHoverPt: null,
  reassigningHitter: false,
  hoverTrackId: null,
  uploading: false,
  pollTimer: null,
};

const els = {
  openLoadBtn: document.getElementById("openLoadBtn"),
  emptyLoadBtn: document.getElementById("emptyLoadBtn"),
  closeLoadBtn: document.getElementById("closeLoadBtn"),
  loadDialog: document.getElementById("loadDialog"),
  uploadForm: document.getElementById("uploadForm"),
  uploadDropzone: document.getElementById("uploadDropzone"),
  selectedFileName: document.getElementById("selectedFileName"),
  uploadSubmitBtn: document.getElementById("uploadSubmitBtn"),
  videoFileInput: document.getElementById("videoFileInput"),
  localPathForm: document.getElementById("localPathForm"),
  localPathInput: document.getElementById("localPathInput"),
  localSubmitBtn: document.getElementById("localSubmitBtn"),
  loadError: document.getElementById("loadError"),
  sessionSelect: document.getElementById("sessionSelect"),
  statusBadge: document.getElementById("statusBadge"),
  statusText: document.getElementById("statusText"),
  analysisProgress: document.getElementById("analysisProgress"),
  calibrateNetBtn: document.getElementById("calibrateNetBtn"),
  netHeightSelect: document.getElementById("netHeightSelect"),
  calScaleBadge: document.getElementById("calScaleBadge"),
  reassignHitterBtn: document.getElementById("reassignHitterBtn"),
  modeHint: document.getElementById("modeHint"),
  cancelModeBtn: document.getElementById("cancelModeBtn"),
  shortcutLegend: document.getElementById("shortcutLegend"),
  videoStage: document.getElementById("videoStage"),
  stageDropOverlay: document.getElementById("stageDropOverlay"),
  video: document.getElementById("playerVideo"),
  canvas: document.getElementById("overlayCanvas"),
  emptyState: document.getElementById("emptyState"),
  stepBack5Btn: document.getElementById("stepBack5Btn"),
  stepBack1Btn: document.getElementById("stepBack1Btn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  playPauseLabel: document.getElementById("playPauseLabel"),
  stepFwd1Btn: document.getElementById("stepFwd1Btn"),
  stepFwd5Btn: document.getElementById("stepFwd5Btn"),
  timelineCanvas: document.getElementById("timelineCanvas"),
  seekSlider: document.getElementById("seekSlider"),
  timeReadout: document.getElementById("timeReadout"),
  keyframesStrip: document.getElementById("keyframesStrip"),
  playsList: document.getElementById("playsList"),
  playsCountBadge: document.getElementById("playsCountBadge"),
  addPlayBtn: document.getElementById("addPlayBtn"),
  metricsList: document.getElementById("metricsList"),
  metricsUnitBadge: document.getElementById("metricsUnitBadge"),
  freezeContactChk: document.getElementById("freezeContactChk"),
  exportMp4Btn: document.getElementById("exportMp4Btn"),
  exportStatus: document.getElementById("exportStatus"),
};

async function fetchCsrf() {
  const res = await fetch("/api/csrf", { credentials: "same-origin" });
  if (!res.ok) throw new Error("Failed to fetch CSRF token");
  const data = await res.json();
  state.csrfToken = data.csrf_token;
}

async function apiPostJson(url, payload) {
  if (!state.csrfToken) await fetchCsrf();
  const res = await fetch(url, {
    method: "POST",
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": state.csrfToken,
    },
    body: JSON.stringify(payload),
  });
  if (res.status === 403) {
    await fetchCsrf();
    const retry = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": state.csrfToken,
      },
      body: JSON.stringify(payload),
    });
    if (!retry.ok) {
      const err = await retry.json().catch(() => ({}));
      throw new Error(err.detail || "Request failed");
    }
    return retry.json();
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

function getActivePlay() {
  if (!state.session || !state.session.plays.length) return null;
  return state.session.plays.find((p) => p.play_id === state.selectedPlayId) || state.session.plays[0];
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const mins = Math.floor(safe / 60);
  const secs = (safe % 60).toFixed(1).padStart(4, "0");
  return `${mins}:${secs}`;
}

function getVideoDisplayRect() {
  const vw = els.video.videoWidth || (state.session ? state.session.metadata.width : 1280);
  const vh = els.video.videoHeight || (state.session ? state.session.metadata.height : 720);
  const cw = els.canvas.clientWidth;
  const ch = els.canvas.clientHeight;
  if (!vw || !vh || !cw || !ch) return { ox: 0, oy: 0, dw: cw, dh: ch, scale: 1 };
  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  return { ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw, dh, scale };
}

function toCanvasPt(pt) {
  const r = getVideoDisplayRect();
  return { x: r.ox + pt.x * r.scale, y: r.oy + pt.y * r.scale };
}

function toVideoPt(clientX, clientY) {
  const rect = els.canvas.getBoundingClientRect();
  const cx = clientX - rect.left;
  const cy = clientY - rect.top;
  const r = getVideoDisplayRect();
  return {
    x: Math.max(0, (cx - r.ox) / r.scale),
    y: Math.max(0, (cy - r.oy) / r.scale),
  };
}

function tapeYAt(cal, x) {
  const dx = cal.right_point.x - cal.left_point.x;
  if (Math.abs(dx) < 1e-6) return cal.left_point.y;
  const t = (x - cal.left_point.x) / dx;
  return cal.left_point.y + t * (cal.right_point.y - cal.left_point.y);
}

function currentFrameIdx() {
  const fps = state.session ? state.session.metadata.fps : 30;
  return Math.max(0, Math.round(els.video.currentTime * fps));
}

function syncControlsDisabledState() {
  const hasSession = Boolean(state.session);
  const hasPlay = Boolean(getActivePlay());

  els.calibrateNetBtn.disabled = !hasSession;
  els.netHeightSelect.disabled = !hasSession;
  els.reassignHitterBtn.disabled = !hasPlay;
  els.stepBack5Btn.disabled = !hasSession;
  els.stepBack1Btn.disabled = !hasSession;
  els.playPauseBtn.disabled = !hasSession;
  els.stepFwd1Btn.disabled = !hasSession;
  els.stepFwd5Btn.disabled = !hasSession;
  els.seekSlider.disabled = !hasSession;
  els.addPlayBtn.disabled = !hasSession;
  els.exportMp4Btn.disabled = !hasPlay;
}

function updateCalScaleBadge() {
  els.calScaleBadge.classList.remove("is-calibrated");
  if (!state.session) {
    els.calScaleBadge.textContent = "Unloaded";
    els.calScaleBadge.title = "Load a video to calibrate the net top tape.";
    els.metricsUnitBadge.textContent = "2D Camera Plane";
    return;
  }
  const cal = state.session.calibration;
  if (cal && cal.is_manual) {
    const spanPx = Math.hypot(
      cal.right_point.x - cal.left_point.x,
      cal.right_point.y - cal.left_point.y
    );
    const cmPerPx = spanPx > 1 ? 900.0 / spanPx : 0;
    els.calScaleBadge.classList.add("is-calibrated");
    els.calScaleBadge.textContent = `Calibrated · ${cmPerPx.toFixed(2)} cm/px`;
    els.calScaleBadge.title = "2-click antenna calibration active (9.0m regulation net span).";
    els.metricsUnitBadge.textContent = "Calibrated Net Plane";
  } else {
    els.calScaleBadge.textContent = "Auto Tape · px";
    els.calScaleBadge.title = "Click 'Calibrate Net (2 Clicks)' on left & right antennas to convert px to cm.";
    els.metricsUnitBadge.textContent = "2D Camera Plane";
  }
}

function getPlayMechanicsRows(play) {
  if (!play) return [];
  const hm = play.hitter_metrics;
  const traj = play.set_trajectory;
  const blk = play.block_summary;
  const setPeakVal = traj
    ? (traj.peak_height_cm !== null
        ? `${traj.peak_height_cm >= 0 ? "+" : ""}${traj.peak_height_cm.toFixed(0)} cm`
        : `${traj.peak_height_px >= 0 ? "+" : ""}${traj.peak_height_px.toFixed(0)} px`)
    : "N/A";
  const contactHtVal = hm && hm.contact_height_px !== null
    ? (hm.contact_height_cm !== null
        ? `${hm.contact_height_cm >= 0 ? "+" : ""}${hm.contact_height_cm.toFixed(0)} cm (${hm.contact_height_px >= 0 ? "+" : ""}${hm.contact_height_px.toFixed(0)}px)`
        : `${hm.contact_height_px >= 0 ? "+" : ""}${hm.contact_height_px.toFixed(0)} px`)
    : "N/A";
  const blockVal = blk.seam_open
    ? (blk.seam_gap_cm !== null ? `Open (${blk.seam_gap_cm.toFixed(0)} cm)` : "Open")
    : (blk.blockers.length >= 2 ? "Closed" : (blk.blockers.length === 1 ? "Single Block" : "No Block"));

  return [
    {
      label: "Attack Zone",
      value: play.attack_zone,
      desc: "Net section where the hit occurs (Zone 4 Left, Zone 3 Middle, Zone 2 Right).",
      color: "#38bdf8",
      accentClass: "accent-cyan",
    },
    {
      label: "Set Tempo",
      value: traj ? `${traj.tempo_s.toFixed(2)} s` : "N/A",
      desc: "Flight time from setter release to hitter ball contact.",
      color: "#38bdf8",
      accentClass: "accent-cyan",
    },
    {
      label: "Set Peak",
      value: setPeakVal,
      desc: "Highest point of the set arc measured vertically above the net top tape.",
      color: "#38bdf8",
      accentClass: "accent-cyan",
    },
    {
      label: "Contact Height",
      value: contactHtVal,
      desc: "Ball height above the net top tape at contact (set 2-click net endpoints for cm).",
      color: "#facc15",
      accentClass: "accent-gold",
      isApprox: Boolean(hm && hm.is_back_row_approximate),
    },
    {
      label: "Arm Extension (2D)",
      value: hm && hm.arm_extension_deg !== null ? `${hm.arm_extension_deg.toFixed(0)}°` : "N/A",
      desc: "Shoulder-elbow-wrist angle at contact in the camera plane (closer to 180° = fuller reach).",
      color: "#4ade80",
      accentClass: "accent-green",
    },
    {
      label: "Torso Tilt (2D)",
      value: hm && hm.torso_tilt_deg !== null ? `${hm.torso_tilt_deg.toFixed(0)}°` : "N/A",
      desc: "Upper-body lean/arch from vertical at contact in the camera plane.",
      color: "#4ade80",
      accentClass: "accent-green",
    },
    {
      label: "Jump Timing",
      value: hm ? hm.jump_timing_label : "N/A",
      desc: "Whether ball contact happens at peak jump height or before/after the apex.",
      color: "#4ade80",
      accentClass: "accent-green",
    },
    {
      label: "Block Seam",
      value: blockVal,
      desc: "Spacing between opposing blockers' hands above the net at contact.",
      color: "#fb7185",
      accentClass: "accent-rose",
    },
  ];
}

function strokeLineWithHalo(ctx, x1, y1, x2, y2, color, width, dash = []) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash(dash);
  ctx.strokeStyle = "rgba(2, 6, 23, 0.72)";
  ctx.lineWidth = width + 2.6;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function drawTopLeftMechanicsHud(ctx, play) {
  const rows = getPlayMechanicsRows(play);
  if (!rows.length) return;
  const r = getVideoDisplayRect();
  const vw = els.video.videoWidth || (state.session ? state.session.metadata.width : 1280);
  const padX = 12;
  const padY = 10;
  const lineH = 18;
  const headerH = 22;

  ctx.font = "600 11px system-ui, sans-serif";
  let maxLabelW = 0;
  let maxValW = 0;
  for (const row of rows) {
    maxLabelW = Math.max(maxLabelW, ctx.measureText(row.label).width);
    maxValW = Math.max(maxValW, ctx.measureText(row.value).width);
  }
  const maxAvailW = Math.max(180, els.canvas.clientWidth - 20);
  const boxW = Math.min(maxAvailW, Math.max(244, Math.ceil(maxLabelW + maxValW + padX * 2 + 30)));
  const boxH = padY * 2 + headerH + rows.length * lineH;

  // Dock HUD on the opposite side of the attack so it never covers the hitter/contact/blockers
  const contactX = play.hitter_metrics && play.hitter_metrics.contact_point
    ? play.hitter_metrics.contact_point.x
    : null;
  const attackOnLeft = contactX !== null
    ? contactX < vw * 0.5
    : /left|zone\s*4/i.test(play.attack_zone || "");
  const leftEdge = Math.max(10, Math.round(r.ox) + 12);
  const rightEdge = Math.min(els.canvas.clientWidth - boxW - 10, Math.round(r.ox + r.dw) - boxW - 12);
  const x = attackOnLeft ? Math.max(10, rightEdge) : leftEdge;
  const y = Math.max(10, Math.round(r.oy) + 12);

  ctx.save();
  ctx.fillStyle = "rgba(9, 13, 22, 0.86)";
  ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.roundRect(x, y, boxW, boxH, 9);
  ctx.fill();
  ctx.stroke();

  ctx.font = "700 10.5px system-ui, sans-serif";
  ctx.fillStyle = "#38bdf8";
  ctx.textAlign = "left";
  ctx.fillText(`PLAY #${play.index} MECHANICS`, x + padX, y + padY + 10);

  ctx.font = "600 10px ui-monospace, monospace";
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "right";
  ctx.fillText(play.attack_zone, x + boxW - padX, y + padY + 10);

  ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x + padX, y + padY + headerH - 4);
  ctx.lineTo(x + boxW - padX, y + padY + headerH - 4);
  ctx.stroke();

  rows.forEach((row, idx) => {
    const ry = y + padY + headerH + 12 + idx * lineH;
    ctx.fillStyle = row.color;
    ctx.beginPath();
    ctx.arc(x + padX + 3, ry - 3.5, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = "500 11px system-ui, sans-serif";
    ctx.fillStyle = "#cbd5e1";
    ctx.textAlign = "left";
    ctx.fillText(row.label, x + padX + 10, ry);

    ctx.font = "700 11px ui-monospace, monospace";
    ctx.fillStyle = row.color;
    ctx.textAlign = "right";
    ctx.fillText(row.value, x + boxW - padX, ry);
  });
  ctx.restore();
}

function drawBadge(ctx, text, x, y, borderCol) {
  ctx.save();
  ctx.font = "600 11.5px system-ui, sans-serif";
  const padX = 8;
  const metrics = ctx.measureText(text);
  const w = metrics.width + padX * 2;
  const h = 22;
  const bx = Math.max(6, Math.min(els.canvas.clientWidth - w - 6, x));
  const by = Math.max(24, Math.min(els.canvas.clientHeight - 6, y));
  ctx.fillStyle = "rgba(9, 13, 22, 0.9)";
  ctx.strokeStyle = borderCol;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.roundRect(bx, by - h, w, h, 5);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f8fafc";
  ctx.textAlign = "left";
  ctx.fillText(text, bx + padX, by - 7);
  ctx.restore();
}

function findNearestAttackingTrackId(play, pt, frameIdx) {
  if (!play || !pt) return null;
  let bestId = null;
  let bestDist = 140;
  for (const track of play.tracks) {
    if (track.court_side !== "attacking") continue;
    const pf = track.frames.find((f) => Math.abs(f.frame_idx - frameIdx) <= 2);
    if (!pf) continue;
    const cx = (pf.bbox[0] + pf.bbox[2]) / 2;
    const cy = (pf.bbox[1] + pf.bbox[3]) / 2;
    const d = Math.hypot(cx - pt.x, cy - pt.y);
    if (d < bestDist) {
      bestDist = d;
      bestId = track.track_id;
    }
  }
  return bestId;
}

function syncActiveKeyframeHighlight(frameIdx) {
  const cards = els.keyframesStrip.querySelectorAll(".kf-card");
  for (const card of cards) {
    const targetFrame = Number(card.dataset.frameIdx);
    const isActive = Number.isFinite(targetFrame) && Math.abs(frameIdx - targetFrame) <= 2;
    card.classList.toggle("active", isActive);
    if (isActive) {
      card.setAttribute("aria-current", "step");
    } else {
      card.removeAttribute("aria-current");
    }
  }
}

function renderTimeline() {
  const canvas = els.timelineCanvas;
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (!cw || !ch) return;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  const pad = 7;
  const usableW = Math.max(1, cw - pad * 2);
  const trackY = Math.round(ch / 2) - 3;
  const trackH = 6;

  // Base track rail
  ctx.fillStyle = "rgba(148, 163, 184, 0.16)";
  ctx.beginPath();
  ctx.roundRect(pad, trackY, usableW, trackH, 3);
  ctx.fill();

  if (!state.session) return;
  const duration = Math.max(0.001, state.session.metadata.duration_s || 1);
  const activePlay = getActivePlay();

  // Play segments
  for (const play of state.session.plays) {
    const x0 = pad + Math.max(0, Math.min(1, play.start_time_s / duration)) * usableW;
    const x1 = pad + Math.max(0, Math.min(1, play.end_time_s / duration)) * usableW;
    const w = Math.max(4, x1 - x0);
    const isSel = activePlay && play.play_id === activePlay.play_id;
    ctx.fillStyle = isSel ? "rgba(56, 189, 248, 0.42)" : "rgba(56, 189, 248, 0.18)";
    ctx.beginPath();
    ctx.roundRect(x0, trackY - (isSel ? 2 : 0), w, trackH + (isSel ? 4 : 0), 3);
    ctx.fill();
  }

  // Played fill
  const progressRatio = Math.max(0, Math.min(1, els.video.currentTime / duration));
  if (progressRatio > 0) {
    ctx.fillStyle = "rgba(56, 189, 248, 0.78)";
    ctx.beginPath();
    ctx.roundRect(pad, trackY, Math.max(3, progressRatio * usableW), trackH, 3);
    ctx.fill();
  }

  // Keyframe ticks for active play
  if (activePlay) {
    const colorMap = {
      set_release: "#38bdf8",
      max_load: "#4ade80",
      contact: "#facc15",
    };
    for (const km of activePlay.keyframes) {
      const kx = pad + Math.max(0, Math.min(1, km.timestamp_s / duration)) * usableW;
      ctx.fillStyle = colorMap[km.name] || "#f8fafc";
      ctx.fillRect(kx - 1.25, trackY - 5, 2.5, trackH + 10);
    }
  }
}

function renderOverlay() {
  const dpr = window.devicePixelRatio || 1;
  const cw = els.canvas.clientWidth;
  const ch = els.canvas.clientHeight;
  if (els.canvas.width !== Math.round(cw * dpr) || els.canvas.height !== Math.round(ch * dpr)) {
    els.canvas.width = Math.round(cw * dpr);
    els.canvas.height = Math.round(ch * dpr);
  }
  const ctx = els.canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  renderTimeline();

  if (!state.session) return;
  const cal = state.session.calibration;
  const frameIdx = currentFrameIdx();
  els.timeReadout.textContent = `${els.video.currentTime.toFixed(2)}s · f${frameIdx}`;
  syncActiveKeyframeHighlight(frameIdx);

  // Net top tape reference
  if (state.overlays.netTape && cal) {
    const lp = toCanvasPt(cal.left_point);
    const rp = toCanvasPt(cal.right_point);
    const tapeCol = cal.is_manual ? "#38bdf8" : "rgba(248, 250, 252, 0.85)";
    strokeLineWithHalo(ctx, lp.x, lp.y, rp.x, rp.y, tapeCol, 2.4, cal.is_manual ? [] : [6, 4]);
    for (const p of [lp, rp]) {
      ctx.fillStyle = "#090d16";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Live 2-click Net Calibration preview
  if (state.calibratingNet) {
    if (state.calFirstPoint) {
      const p1 = toCanvasPt(state.calFirstPoint);
      ctx.fillStyle = "#facc15";
      ctx.beginPath();
      ctx.arc(p1.x, p1.y, 6, 0, Math.PI * 2);
      ctx.fill();

      if (state.calHoverPt) {
        const p2 = toCanvasPt(state.calHoverPt);
        strokeLineWithHalo(ctx, p1.x, p1.y, p2.x, p2.y, "#facc15", 2.2, [6, 4]);
        const pxSpan = Math.hypot(
          state.calHoverPt.x - state.calFirstPoint.x,
          state.calHoverPt.y - state.calFirstPoint.y
        );
        drawBadge(ctx, `Span ${Math.round(pxSpan)}px (9.0m)`, p2.x + 12, p2.y - 6, "#facc15");
      }
    } else if (state.calHoverPt) {
      const chPt = toCanvasPt(state.calHoverPt);
      ctx.save();
      ctx.strokeStyle = "rgba(250, 204, 21, 0.85)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(chPt.x, chPt.y, 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  const play = getActivePlay();
  if (!play) return;

  // 1. Setting Path Arc
  if (state.overlays.setArc && play.set_trajectory && play.set_trajectory.arc_points.length) {
    const traj = play.set_trajectory;
    const vis = traj.arc_points.filter((pt) => pt.frame_idx <= frameIdx);
    if (vis.length >= 2) {
      const startPt = toCanvasPt(vis[0]);
      const endPt = toCanvasPt(vis[vis.length - 1]);
      const grad = ctx.createLinearGradient(startPt.x, startPt.y, endPt.x, endPt.y);
      grad.addColorStop(0, "#38bdf8");
      grad.addColorStop(1, "#facc15");

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "rgba(2, 6, 23, 0.65)";
      ctx.lineWidth = 5.5;
      ctx.beginPath();
      vis.forEach((pt, i) => {
        const cp = toCanvasPt(pt);
        if (i === 0) ctx.moveTo(cp.x, cp.y);
        else ctx.lineTo(cp.x, cp.y);
      });
      ctx.stroke();

      ctx.strokeStyle = grad;
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      vis.forEach((pt, i) => {
        const cp = toCanvasPt(pt);
        if (i === 0) ctx.moveTo(cp.x, cp.y);
        else ctx.lineTo(cp.x, cp.y);
      });
      ctx.stroke();
      ctx.restore();
    }
    vis.forEach((pt, idx) => {
      if (idx % 4 === 0 || idx === vis.length - 1) {
        const cp = toCanvasPt(pt);
        ctx.fillStyle = pt.is_interpolated ? "rgba(250, 204, 21, 0.55)" : "#facc15";
        ctx.beginPath();
        ctx.arc(cp.x, cp.y, 3.8, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    if (state.overlays.inlineStats && frameIdx >= traj.peak_frame) {
      const pk = toCanvasPt(traj.peak_point);
      drawBadge(ctx, `Set Arc · ${traj.tempo_s.toFixed(2)}s`, pk.x - 45, pk.y - 12, "#38bdf8");
    }
  }

  // 2. Hitter Pose Skeleton (or all attacking candidates during Reassign mode)
  if (state.overlays.pose || state.reassigningHitter) {
    if (state.reassigningHitter && state.calHoverPt) {
      state.hoverTrackId = findNearestAttackingTrackId(play, state.calHoverPt, frameIdx);
    } else {
      state.hoverTrackId = null;
    }

    for (const track of play.tracks) {
      const isHitter = track.track_id === play.hitter_track_id;
      const isHovered = state.reassigningHitter && track.track_id === state.hoverTrackId;
      if (!isHitter && !state.reassigningHitter) continue;
      if (state.reassigningHitter && track.court_side !== "attacking") continue;
      const pf = track.frames.find((f) => Math.abs(f.frame_idx - frameIdx) <= 2);
      if (!pf) continue;

      const dom = play.hitter_metrics ? play.hitter_metrics.dominant_hand : "right";
      const lineW = isHitter || isHovered ? 3 : 2;
      for (const [a, b] of COCO_EDGES) {
        const ka = pf.keypoints[a];
        const kb = pf.keypoints[b];
        if (!ka || !kb || ka[2] < 0.25 || kb[2] < 0.25) continue;
        const pa = toCanvasPt({ x: ka[0], y: ka[1] });
        const pb = toCanvasPt({ x: kb[0], y: kb[1] });
        const isHitArm = isHitter && (
          (dom === "right" && ((a === 6 && b === 8) || (a === 8 && b === 10))) ||
          (dom === "left" && ((a === 5 && b === 7) || (a === 7 && b === 9)))
        );
        const col = isHovered
          ? "#facc15"
          : (isHitArm ? "#facc15" : (isHitter ? "#4ade80" : "rgba(148, 163, 184, 0.72)"));
        strokeLineWithHalo(ctx, pa.x, pa.y, pb.x, pb.y, col, lineW);
      }

      if (isHitter || isHovered) {
        for (const kp of pf.keypoints) {
          if (!kp || kp[2] < 0.25) continue;
          const pk = toCanvasPt({ x: kp[0], y: kp[1] });
          ctx.fillStyle = isHovered ? "#facc15" : "#4ade80";
          ctx.beginPath();
          ctx.arc(pk.x, pk.y, 3.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (state.reassigningHitter) {
        const topLeft = toCanvasPt({ x: pf.bbox[0], y: pf.bbox[1] });
        const label = isHovered
          ? `Click to assign Track #${track.track_id}`
          : (isHitter ? `Active · Track #${track.track_id}` : `Track #${track.track_id}`);
        drawBadge(
          ctx,
          label,
          topLeft.x,
          topLeft.y - 4,
          isHovered ? "#facc15" : (isHitter ? "#4ade80" : "#94a3b8")
        );
      }
    }
  }

  // 3. Contact Point & Posture Callout
  const contactKm = play.keyframes.find((k) => k.name === "contact");
  if (state.overlays.contact && contactKm && Math.abs(frameIdx - contactKm.frame_idx) <= 3 && play.hitter_metrics) {
    const hm = play.hitter_metrics;
    if (hm.contact_point) {
      const cp = toCanvasPt(hm.contact_point);
      const tp = toCanvasPt({ x: hm.contact_point.x, y: tapeYAt(cal, hm.contact_point.x) });

      strokeLineWithHalo(ctx, cp.x, cp.y, tp.x, tp.y, "#facc15", 2.2);
      strokeLineWithHalo(ctx, cp.x - 6, cp.y, cp.x + 6, cp.y, "#facc15", 1.8);
      strokeLineWithHalo(ctx, tp.x - 6, tp.y, tp.x + 6, tp.y, "#facc15", 1.8);

      ctx.save();
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(cp.x, cp.y, 13, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#facc15";
      ctx.beginPath();
      ctx.arc(cp.x, cp.y, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      if (state.overlays.inlineStats) {
        const htText = hm.contact_height_cm !== null
          ? `${hm.contact_height_cm >= 0 ? "+" : ""}${hm.contact_height_cm.toFixed(0)}cm`
          : `${hm.contact_height_px >= 0 ? "+" : ""}${hm.contact_height_px.toFixed(0)}px`;
        const dispRect = getVideoDisplayRect();
        const placeLeft = cp.x > dispRect.ox + dispRect.dw * 0.65;
        const badgeX = placeLeft ? Math.max(10, cp.x - 190) : cp.x + 16;
        drawBadge(ctx, `Contact ${htText} above tape`, badgeX, cp.y - 4, "#facc15");
        if (hm.arm_extension_deg !== null) {
          drawBadge(ctx, `Arm ${hm.arm_extension_deg.toFixed(0)}° · ${hm.jump_timing_label}`, badgeX, cp.y + 22, "#4ade80");
        }
      }
    }
  }

  // 4. Opposing Blockers & Open Seam
  if (state.overlays.blockers && contactKm && Math.abs(frameIdx - contactKm.frame_idx) <= 4 && play.block_summary) {
    for (const blk of play.block_summary.blockers) {
      const tl = toCanvasPt({ x: blk.bbox[0], y: blk.bbox[1] });
      const br = toCanvasPt({ x: blk.bbox[2], y: blk.bbox[3] });
      ctx.save();
      ctx.fillStyle = "rgba(251, 113, 133, 0.1)";
      ctx.fillRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
      ctx.strokeStyle = "#fb7185";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
      ctx.restore();

      const reach = toCanvasPt(blk.top_reach_point);
      ctx.fillStyle = "#fb7185";
      ctx.beginPath();
      ctx.arc(reach.x, reach.y, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (play.block_summary.seam_open && play.block_summary.seam_interval_x) {
      const [sx1, sx2] = play.block_summary.seam_interval_x;
      const sy = tapeYAt(cal, (sx1 + sx2) / 2) - 14;
      const p1 = toCanvasPt({ x: sx1, y: sy });
      const p2 = toCanvasPt({ x: sx2, y: sy });
      strokeLineWithHalo(ctx, p1.x, p1.y, p2.x, p2.y, "#fb7185", 3);
      if (state.overlays.inlineStats) {
        drawBadge(ctx, "Block Seam Open", p1.x, p1.y - 8, "#fb7185");
      }
    }
  }

  // 5. Play Mechanics HUD on top-left
  if (state.overlays.hud) {
    drawTopLeftMechanicsHud(ctx, play);
  }
}

function jumpToFrame(frameIdx) {
  if (!state.session) return;
  const fps = state.session.metadata.fps || 30;
  els.video.pause();
  els.video.currentTime = Math.max(0, Math.min(state.session.metadata.duration_s, frameIdx / fps));
  updatePlayPauseLabel();
  renderOverlay();
}

function renderKeyframes(play) {
  els.keyframesStrip.replaceChildren();
  if (!play || !play.keyframes.length) {
    const empty = document.createElement("div");
    empty.className = "keyframe-placeholder";
    empty.textContent = "Select or load a play to jump to Set Release, Max Load, and Ball Contact.";
    els.keyframesStrip.appendChild(empty);
    return;
  }

  play.keyframes.forEach((km, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "kf-card";
    btn.dataset.frameIdx = String(km.frame_idx);
    btn.dataset.moment = km.name;
    btn.addEventListener("click", () => jumpToFrame(km.frame_idx));

    const thumbWrap = document.createElement("div");
    thumbWrap.className = "kf-thumb-wrap";

    const img = document.createElement("img");
    img.className = "kf-thumb";
    img.alt = km.label;
    if (km.thumbnail_rel_path && state.session) {
      img.src = `/api/sessions/${encodeURIComponent(state.session.session_id)}/thumbnails/${encodeURIComponent(km.thumbnail_rel_path)}`;
    } else {
      img.classList.add("hidden");
    }
    img.addEventListener("error", () => img.classList.add("hidden"));

    const stepBadge = document.createElement("span");
    stepBadge.className = "kf-step";
    stepBadge.textContent = String(index + 1);

    thumbWrap.appendChild(img);
    thumbWrap.appendChild(stepBadge);
    btn.appendChild(thumbWrap);

    const info = document.createElement("div");
    info.className = "kf-info";
    const title = document.createElement("div");
    title.className = "kf-title";
    title.textContent = km.label;
    const meta = document.createElement("div");
    meta.className = "kf-meta";
    meta.textContent = `${km.timestamp_s.toFixed(2)}s · f${km.frame_idx}`;
    info.appendChild(title);
    info.appendChild(meta);
    btn.appendChild(info);

    els.keyframesStrip.appendChild(btn);
  });

  syncActiveKeyframeHighlight(currentFrameIdx());
}

function renderMetrics(play) {
  els.metricsList.replaceChildren();
  const rows = getPlayMechanicsRows(play);
  if (!rows.length) {
    const item = document.createElement("div");
    item.className = "metric-item accent-cyan";
    const head = document.createElement("div");
    head.className = "metric-head";
    const dt = document.createElement("dt");
    dt.className = "metric-label";
    dt.textContent = "Status";
    const dd = document.createElement("dd");
    dd.className = "metric-val";
    dd.textContent = "Select a play";
    head.appendChild(dt);
    head.appendChild(dd);
    item.appendChild(head);
    els.metricsList.appendChild(item);
    return;
  }

  for (const row of rows) {
    const item = document.createElement("div");
    item.className = `metric-item ${row.accentClass}`;

    const head = document.createElement("div");
    head.className = "metric-head";

    const dt = document.createElement("dt");
    dt.className = "metric-label";
    dt.textContent = row.label;

    const valWrap = document.createElement("div");
    valWrap.className = "metric-val-wrap";

    if (row.isApprox) {
      const approx = document.createElement("span");
      approx.className = "approx-badge";
      approx.title = "Hitter took off behind the 3m line; height above tape is approximate.";
      approx.textContent = "~Back-row";
      valWrap.appendChild(approx);
    }

    const dd = document.createElement("dd");
    dd.className = "metric-val";
    dd.textContent = row.value;
    valWrap.appendChild(dd);

    head.appendChild(dt);
    head.appendChild(valWrap);

    const desc = document.createElement("div");
    desc.className = "metric-desc";
    desc.textContent = row.desc;

    item.appendChild(head);
    item.appendChild(desc);
    els.metricsList.appendChild(item);
  }
}

function renderPlaysList() {
  els.playsList.replaceChildren();
  const count = state.session ? state.session.plays.length : 0;
  els.playsCountBadge.textContent = String(count);

  if (!state.session || !state.session.plays.length) {
    const empty = document.createElement("div");
    empty.className = "plays-empty";
    empty.textContent = state.session
      ? "Scanning video for set-and-hit plays..."
      : "Load a video to inspect detected plays.";
    els.playsList.appendChild(empty);
    return;
  }

  for (const play of state.session.plays) {
    const isActive = play.play_id === state.selectedPlayId;
    const item = document.createElement("div");
    item.className = "play-item" + (isActive ? " active" : "");
    item.setAttribute("role", "listitem");

    const selectBtn = document.createElement("button");
    selectBtn.type = "button";
    selectBtn.className = "play-select-btn";
    if (isActive) selectBtn.setAttribute("aria-current", "true");
    selectBtn.addEventListener("click", () => selectPlay(play.play_id, true));

    const titleRow = document.createElement("div");
    titleRow.className = "play-title-row";

    const title = document.createElement("span");
    title.className = "play-title";
    title.textContent = `Play #${play.index}`;

    const timeTag = document.createElement("span");
    timeTag.className = "play-time-tag";
    timeTag.textContent = formatTime(play.start_time_s);

    titleRow.appendChild(title);
    titleRow.appendChild(timeTag);

    const sub = document.createElement("div");
    sub.className = "play-sub";

    const zoneChip = document.createElement("span");
    zoneChip.className = "zone-chip";
    zoneChip.textContent = play.attack_zone;
    sub.appendChild(zoneChip);

    const metaParts = [];
    if (play.set_trajectory) {
      metaParts.push(`${play.set_trajectory.tempo_s.toFixed(2)}s`);
    }
    const hm = play.hitter_metrics;
    if (hm && hm.contact_height_cm !== null) {
      metaParts.push(`${hm.contact_height_cm >= 0 ? "+" : ""}${hm.contact_height_cm.toFixed(0)}cm`);
    } else if (hm && hm.contact_height_px !== null) {
      metaParts.push(`${hm.contact_height_px >= 0 ? "+" : ""}${hm.contact_height_px.toFixed(0)}px`);
    }
    if (metaParts.length) {
      const metaSpan = document.createElement("span");
      metaSpan.textContent = metaParts.join(" · ");
      sub.appendChild(metaSpan);
    }

    selectBtn.appendChild(titleRow);
    selectBtn.appendChild(sub);

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "play-delete";
    delBtn.title = `Remove Play #${play.index}`;
    delBtn.setAttribute("aria-label", `Remove Play #${play.index}`);
    delBtn.textContent = "×";
    delBtn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      delBtn.disabled = true;
      try {
        const updated = await apiPostJson(
          `/api/sessions/${encodeURIComponent(state.session.session_id)}/plays/${encodeURIComponent(play.play_id)}/delete`,
          {}
        );
        applySession(updated, false);
      } catch (err) {
        delBtn.disabled = false;
        els.exportStatus.classList.add("is-error");
        els.exportStatus.textContent = err.message;
      }
    });

    item.appendChild(selectBtn);
    item.appendChild(delBtn);
    els.playsList.appendChild(item);
  }
}

function selectPlay(playId, seekToStart) {
  state.selectedPlayId = playId;
  const play = getActivePlay();
  renderPlaysList();
  renderKeyframes(play);
  renderMetrics(play);
  syncControlsDisabledState();
  if (play && seekToStart) {
    const startKm = play.keyframes.find((k) => k.name === "set_release");
    jumpToFrame(startKm ? startKm.frame_idx : play.start_frame);
  } else {
    renderOverlay();
  }
}

function updateStatusPill(sess) {
  els.statusBadge.className = "status-pill";
  if (!sess) {
    els.statusText.textContent = "No Video Loaded";
    els.analysisProgress.classList.add("hidden");
    els.analysisProgress.value = 0;
    return;
  }
  if (sess.status === "ready") {
    els.statusBadge.classList.add("status-ready");
    els.statusText.textContent = `Ready · ${sess.plays.length} play${sess.plays.length === 1 ? "" : "s"}`;
    els.analysisProgress.classList.add("hidden");
    els.analysisProgress.value = 100;
  } else if (sess.status === "error") {
    els.statusBadge.classList.add("status-error");
    els.statusText.textContent = sess.error_message || "Analysis Error";
    els.analysisProgress.classList.add("hidden");
  } else {
    const pct = Math.max(0, Math.min(100, Math.round(sess.progress_pct || 0)));
    els.statusBadge.classList.add("status-busy");
    els.statusText.textContent = `${sess.status.toUpperCase()} (${pct}%)`;
    els.analysisProgress.classList.remove("hidden");
    els.analysisProgress.value = pct;
  }
}

function applySession(sess, resetVideoSrc) {
  const prevId = state.session ? state.session.session_id : null;
  state.session = sess;
  updateStatusPill(sess);
  updateCalScaleBadge();
  els.netHeightSelect.value = String(sess.calibration.net_height_m || 2.43);

  if (resetVideoSrc || prevId !== sess.session_id) {
    els.emptyState.classList.add("hidden");
    els.video.src = `/api/sessions/${encodeURIComponent(sess.session_id)}/video`;
    els.seekSlider.max = String(Math.max(1, sess.metadata.duration_s));
    state.selectedPlayId = sess.plays.length ? sess.plays[0].play_id : null;
    els.exportStatus.classList.remove("is-error");
    els.exportStatus.textContent = "";
  } else if (!state.selectedPlayId && sess.plays.length) {
    state.selectedPlayId = sess.plays[0].play_id;
  } else if (state.selectedPlayId && !sess.plays.some((p) => p.play_id === state.selectedPlayId)) {
    state.selectedPlayId = sess.plays.length ? sess.plays[0].play_id : null;
  }

  const activePlay = getActivePlay();
  renderPlaysList();
  renderKeyframes(activePlay);
  renderMetrics(activePlay);
  syncControlsDisabledState();
  renderOverlay();

  if (sess.status !== "ready" && sess.status !== "error") {
    schedulePoll(sess.session_id);
  } else if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

function schedulePoll(sessionId) {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { credentials: "same-origin" });
      if (res.ok) {
        const fresh = await res.json();
        applySession(fresh, false);
      }
    } catch {
      schedulePoll(sessionId);
    }
  }, 1200);
}

async function refreshSessionsDropdown() {
  const res = await fetch("/api/sessions", { credentials: "same-origin" });
  if (!res.ok) return;
  const list = await res.json();
  els.sessionSelect.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = list.length ? "Switch session..." : "No saved sessions";
  els.sessionSelect.appendChild(placeholder);

  for (const s of list) {
    const opt = document.createElement("option");
    opt.value = s.session_id;
    opt.textContent = `${s.filename} (${s.plays.length} plays)`;
    if (state.session && state.session.session_id === s.session_id) opt.selected = true;
    els.sessionSelect.appendChild(opt);
  }
}

function updatePlayPauseLabel() {
  const playing = !els.video.paused && !els.video.ended;
  els.playPauseBtn.classList.toggle("is-playing", playing);
  els.playPauseLabel.textContent = playing ? "Pause" : "Play";
}

function stepFrames(delta) {
  if (!state.session) return;
  const fps = state.session.metadata.fps || 30;
  els.video.pause();
  els.video.currentTime = Math.max(0, Math.min(state.session.metadata.duration_s, els.video.currentTime + delta / fps));
  updatePlayPauseLabel();
  renderOverlay();
}

function setInteractiveMode(calNet, reassign) {
  state.calibratingNet = calNet;
  state.calFirstPoint = null;
  state.calHoverPt = null;
  state.reassigningHitter = reassign;
  state.hoverTrackId = null;

  els.calibrateNetBtn.classList.toggle("active-mode", calNet);
  els.reassignHitterBtn.classList.toggle("active-mode", reassign);
  els.canvas.classList.toggle("interactive", calNet || reassign);

  const inMode = calNet || reassign;
  els.modeHint.classList.toggle("hidden", !inMode);
  els.modeHint.classList.toggle("highlight", inMode);
  els.cancelModeBtn.classList.toggle("hidden", !inMode);
  els.shortcutLegend.classList.toggle("hidden", inMode);

  if (calNet) {
    els.modeHint.textContent = "Step 1/2: Click the LEFT antenna top-tape corner on the video.";
  } else if (reassign) {
    els.modeHint.textContent = "Click any attacking player skeleton on the video to set as Active Hitter.";
  } else {
    els.modeHint.textContent = "";
  }
  renderOverlay();
}

function isSupportedVideoFile(file) {
  if (!file || !file.name) return false;
  const lower = file.name.toLowerCase();
  return ALLOWED_VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function updateSelectedFileDisplay(file) {
  if (file) {
    const mb = (file.size / (1024 * 1024)).toFixed(1);
    els.selectedFileName.textContent = `${file.name} (${mb} MB)`;
    els.uploadDropzone.classList.add("has-file");
  } else {
    els.selectedFileName.textContent = "No file selected";
    els.uploadDropzone.classList.remove("has-file");
  }
}

async function submitVideoFile(file) {
  if (state.uploading) return;
  els.loadError.textContent = "";
  if (!file) {
    els.loadError.textContent = "Choose a video file first.";
    return;
  }
  if (!isSupportedVideoFile(file)) {
    els.loadError.textContent = "Unsupported file type. Choose .mp4, .mov, .mkv, or .avi.";
    if (!els.loadDialog.open) els.loadDialog.showModal();
    return;
  }

  state.uploading = true;
  els.uploadSubmitBtn.disabled = true;
  els.uploadSubmitBtn.textContent = "Uploading & Analysing...";
  try {
    if (!state.csrfToken) await fetchCsrf();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("net_height_m", els.netHeightSelect.value);
    const res = await fetch("/api/sessions/upload", {
      method: "POST",
      credentials: "same-origin",
      headers: { "X-CSRF-Token": state.csrfToken },
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      els.loadError.textContent = err.detail || "Upload failed.";
      if (!els.loadDialog.open) els.loadDialog.showModal();
      return;
    }
    const sess = await res.json();
    if (els.loadDialog.open) els.loadDialog.close();
    applySession(sess, true);
    refreshSessionsDropdown();
  } catch (err) {
    els.loadError.textContent = err.message || "Upload failed.";
    if (!els.loadDialog.open) els.loadDialog.showModal();
  } finally {
    state.uploading = false;
    els.uploadSubmitBtn.disabled = false;
    els.uploadSubmitBtn.textContent = "Upload & Analyse";
  }
}

function setupEventListeners() {
  const openModal = () => {
    els.loadError.textContent = "";
    els.loadDialog.showModal();
  };
  els.openLoadBtn.addEventListener("click", openModal);
  els.emptyLoadBtn.addEventListener("click", openModal);
  els.closeLoadBtn.addEventListener("click", () => els.loadDialog.close());
  els.loadDialog.addEventListener("click", (ev) => {
    if (ev.target === els.loadDialog) els.loadDialog.close();
  });

  els.videoFileInput.addEventListener("change", () => {
    updateSelectedFileDisplay(els.videoFileInput.files[0] || null);
  });

  els.uploadDropzone.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    els.uploadDropzone.classList.add("drag-over");
  });
  els.uploadDropzone.addEventListener("dragleave", () => {
    els.uploadDropzone.classList.remove("drag-over");
  });
  els.uploadDropzone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    els.uploadDropzone.classList.remove("drag-over");
    const file = ev.dataTransfer && ev.dataTransfer.files ? ev.dataTransfer.files[0] : null;
    if (file) {
      updateSelectedFileDisplay(file);
      submitVideoFile(file);
    }
  });

  els.videoStage.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    els.videoStage.classList.add("drag-active");
    els.stageDropOverlay.classList.remove("hidden");
  });
  els.videoStage.addEventListener("dragleave", (ev) => {
    if (!els.videoStage.contains(ev.relatedTarget)) {
      els.videoStage.classList.remove("drag-active");
      els.stageDropOverlay.classList.add("hidden");
    }
  });
  els.videoStage.addEventListener("drop", (ev) => {
    ev.preventDefault();
    els.videoStage.classList.remove("drag-active");
    els.stageDropOverlay.classList.add("hidden");
    const file = ev.dataTransfer && ev.dataTransfer.files ? ev.dataTransfer.files[0] : null;
    if (file) {
      submitVideoFile(file);
    }
  });

  els.uploadForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    await submitVideoFile(els.videoFileInput.files[0] || null);
  });

  els.localPathForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    els.loadError.textContent = "";
    const rawPath = els.localPathInput.value.trim();
    if (!rawPath) {
      els.loadError.textContent = "Enter a local file path.";
      return;
    }
    els.localSubmitBtn.disabled = true;
    els.localSubmitBtn.textContent = "Loading Local File...";
    try {
      const sess = await apiPostJson("/api/sessions/local", {
        path: rawPath,
        net_height_m: parseFloat(els.netHeightSelect.value),
      });
      els.loadDialog.close();
      applySession(sess, true);
      refreshSessionsDropdown();
    } catch (err) {
      els.loadError.textContent = err.message;
    } finally {
      els.localSubmitBtn.disabled = false;
      els.localSubmitBtn.textContent = "Analyse Local File";
    }
  });

  els.sessionSelect.addEventListener("change", async () => {
    const sid = els.sessionSelect.value;
    if (!sid) return;
    const res = await fetch(`/api/sessions/${encodeURIComponent(sid)}`, { credentials: "same-origin" });
    if (res.ok) applySession(await res.json(), true);
  });

  document.querySelectorAll(".pill[data-overlay]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-overlay");
      state.overlays[key] = !state.overlays[key];
      const active = Boolean(state.overlays[key]);
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-pressed", String(active));
      renderOverlay();
    });
  });

  els.calibrateNetBtn.addEventListener("click", () => {
    if (!state.session) return;
    setInteractiveMode(!state.calibratingNet, false);
  });

  els.reassignHitterBtn.addEventListener("click", () => {
    if (!getActivePlay()) return;
    setInteractiveMode(false, !state.reassigningHitter);
  });

  els.cancelModeBtn.addEventListener("click", () => {
    setInteractiveMode(false, false);
  });

  els.netHeightSelect.addEventListener("change", async () => {
    if (!state.session) return;
    const cal = state.session.calibration;
    const updated = await apiPostJson(
      `/api/sessions/${encodeURIComponent(state.session.session_id)}/calibration`,
      {
        left_point: cal.left_point,
        right_point: cal.right_point,
        net_height_m: parseFloat(els.netHeightSelect.value),
      }
    );
    applySession(updated, false);
  });

  els.canvas.addEventListener("mousemove", (ev) => {
    if (!state.session || (!state.calibratingNet && !state.reassigningHitter)) return;
    state.calHoverPt = toVideoPt(ev.clientX, ev.clientY);
    renderOverlay();
  });

  els.canvas.addEventListener("mouseleave", () => {
    if (!state.calHoverPt && state.hoverTrackId === null) return;
    state.calHoverPt = null;
    state.hoverTrackId = null;
    renderOverlay();
  });

  els.canvas.addEventListener("click", async (ev) => {
    if (!state.session) return;
    const pt = toVideoPt(ev.clientX, ev.clientY);

    if (state.calibratingNet) {
      if (!state.calFirstPoint) {
        state.calFirstPoint = pt;
        els.modeHint.textContent = "Step 2/2: Now click the RIGHT antenna top-tape corner.";
        renderOverlay();
      } else {
        const leftPt = pt.x >= state.calFirstPoint.x ? state.calFirstPoint : pt;
        const rightPt = pt.x >= state.calFirstPoint.x ? pt : state.calFirstPoint;
        const updated = await apiPostJson(
          `/api/sessions/${encodeURIComponent(state.session.session_id)}/calibration`,
          {
            left_point: leftPt,
            right_point: rightPt,
            net_height_m: parseFloat(els.netHeightSelect.value),
          }
        );
        setInteractiveMode(false, false);
        applySession(updated, false);
      }
      return;
    }

    if (state.reassigningHitter) {
      const play = getActivePlay();
      if (!play) return;
      const bestId = findNearestAttackingTrackId(play, pt, currentFrameIdx());
      if (bestId !== null) {
        const updated = await apiPostJson(
          `/api/sessions/${encodeURIComponent(state.session.session_id)}/plays/${encodeURIComponent(play.play_id)}/hitter`,
          { track_id: bestId }
        );
        setInteractiveMode(false, false);
        applySession(updated, false);
      }
    }
  });

  els.playPauseBtn.addEventListener("click", () => {
    if (!state.session) return;
    if (els.video.paused) els.video.play();
    else els.video.pause();
    updatePlayPauseLabel();
  });
  els.stepBack5Btn.addEventListener("click", () => stepFrames(-5));
  els.stepBack1Btn.addEventListener("click", () => stepFrames(-1));
  els.stepFwd1Btn.addEventListener("click", () => stepFrames(1));
  els.stepFwd5Btn.addEventListener("click", () => stepFrames(5));

  els.seekSlider.addEventListener("input", () => {
    if (!state.session) return;
    els.video.currentTime = parseFloat(els.seekSlider.value);
    renderOverlay();
  });

  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const spd = parseFloat(btn.getAttribute("data-speed"));
      els.video.playbackRate = spd;
      document.querySelectorAll(".speed-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-pressed", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-pressed", "true");
    });
  });

  els.addPlayBtn.addEventListener("click", async () => {
    if (!state.session) return;
    els.addPlayBtn.disabled = true;
    try {
      const updated = await apiPostJson(
        `/api/sessions/${encodeURIComponent(state.session.session_id)}/plays/add`,
        { center_time_s: els.video.currentTime }
      );
      applySession(updated, false);
    } catch (err) {
      els.exportStatus.classList.add("is-error");
      els.exportStatus.textContent = err.message;
    } finally {
      syncControlsDisabledState();
    }
  });

  els.exportMp4Btn.addEventListener("click", async () => {
    const play = getActivePlay();
    if (!state.session || !play) return;
    els.exportMp4Btn.disabled = true;
    els.exportStatus.classList.remove("is-error");
    els.exportStatus.textContent = "Rendering annotated MP4...";
    try {
      const res = await apiPostJson(
        `/api/sessions/${encodeURIComponent(state.session.session_id)}/plays/${encodeURIComponent(play.play_id)}/export`,
        {
          freeze_contact: els.freezeContactChk.checked,
          show_inline_stats: state.overlays.inlineStats,
        }
      );
      els.exportStatus.replaceChildren();
      const link = document.createElement("a");
      link.href = res.download_url;
      link.download = res.filename;
      link.textContent = `Download ${res.filename}`;
      els.exportStatus.appendChild(link);
      link.click();
    } catch (err) {
      els.exportStatus.classList.add("is-error");
      els.exportStatus.textContent = err.message;
    } finally {
      syncControlsDisabledState();
    }
  });

  els.video.addEventListener("timeupdate", () => {
    els.seekSlider.value = String(els.video.currentTime);
    renderOverlay();
  });
  els.video.addEventListener("seeked", renderOverlay);
  els.video.addEventListener("loadedmetadata", renderOverlay);
  els.video.addEventListener("play", () => {
    updatePlayPauseLabel();
    const tick = () => {
      renderOverlay();
      if (!els.video.paused && !els.video.ended) {
        if (typeof els.video.requestVideoFrameCallback === "function") {
          els.video.requestVideoFrameCallback(tick);
        } else {
          requestAnimationFrame(tick);
        }
      }
    };
    tick();
  });
  els.video.addEventListener("pause", updatePlayPauseLabel);
  window.addEventListener("resize", renderOverlay);

  document.addEventListener("keydown", (ev) => {
    if (ev.code === "Escape" && (state.calibratingNet || state.reassigningHitter)) {
      ev.preventDefault();
      setInteractiveMode(false, false);
      return;
    }
    if (ev.target && ["INPUT", "SELECT", "TEXTAREA"].includes(ev.target.tagName)) return;
    if (ev.code === "Space") {
      ev.preventDefault();
      els.playPauseBtn.click();
    } else if (ev.code === "ArrowLeft") {
      ev.preventDefault();
      stepFrames(-1);
    } else if (ev.code === "ArrowRight") {
      ev.preventDefault();
      stepFrames(1);
    } else if (["Digit1", "Digit2", "Digit3"].includes(ev.code)) {
      const play = getActivePlay();
      if (!play) return;
      const idx = parseInt(ev.code.replace("Digit", ""), 10) - 1;
      if (play.keyframes[idx]) jumpToFrame(play.keyframes[idx].frame_idx);
    }
  });
}

async function init() {
  syncControlsDisabledState();
  updateCalScaleBadge();
  setupEventListeners();
  await fetchCsrf();
  await refreshSessionsDropdown();
}

init();

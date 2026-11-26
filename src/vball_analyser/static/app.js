"use strict";

const COCO_EDGES = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];

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
  },
  calibratingNet: false,
  calFirstPoint: null,
  reassigningHitter: false,
  pollTimer: null,
};

const els = {
  openLoadBtn: document.getElementById("openLoadBtn"),
  emptyLoadBtn: document.getElementById("emptyLoadBtn"),
  closeLoadBtn: document.getElementById("closeLoadBtn"),
  loadDialog: document.getElementById("loadDialog"),
  uploadForm: document.getElementById("uploadForm"),
  videoFileInput: document.getElementById("videoFileInput"),
  localPathForm: document.getElementById("localPathForm"),
  localPathInput: document.getElementById("localPathInput"),
  loadError: document.getElementById("loadError"),
  sessionSelect: document.getElementById("sessionSelect"),
  statusBadge: document.getElementById("statusBadge"),
  calibrateNetBtn: document.getElementById("calibrateNetBtn"),
  netHeightSelect: document.getElementById("netHeightSelect"),
  reassignHitterBtn: document.getElementById("reassignHitterBtn"),
  modeHint: document.getElementById("modeHint"),
  videoStage: document.getElementById("videoStage"),
  video: document.getElementById("playerVideo"),
  canvas: document.getElementById("overlayCanvas"),
  emptyState: document.getElementById("emptyState"),
  stepBack5Btn: document.getElementById("stepBack5Btn"),
  stepBack1Btn: document.getElementById("stepBack1Btn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  stepFwd1Btn: document.getElementById("stepFwd1Btn"),
  stepFwd5Btn: document.getElementById("stepFwd5Btn"),
  seekSlider: document.getElementById("seekSlider"),
  timeReadout: document.getElementById("timeReadout"),
  keyframesStrip: document.getElementById("keyframesStrip"),
  playsList: document.getElementById("playsList"),
  addPlayBtn: document.getElementById("addPlayBtn"),
  metricsList: document.getElementById("metricsList"),
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
    },
    {
      label: "Set Tempo",
      value: traj ? `${traj.tempo_s.toFixed(2)} s` : "N/A",
      desc: "Flight time from setter release to hitter ball contact.",
      color: "#38bdf8",
    },
    {
      label: "Set Peak",
      value: setPeakVal,
      desc: "Highest point of the set arc measured vertically above the net top tape.",
      color: "#38bdf8",
    },
    {
      label: "Contact Height",
      value: contactHtVal,
      desc: "Ball height above the net top tape at contact (set 2-click net endpoints for cm).",
      color: "#facc15",
    },
    {
      label: "Arm Extension (2D)",
      value: hm && hm.arm_extension_deg !== null ? `${hm.arm_extension_deg.toFixed(0)}°` : "N/A",
      desc: "Shoulder-elbow-wrist angle at contact in the camera plane (closer to 180° = fuller reach).",
      color: "#4ade80",
    },
    {
      label: "Torso Tilt (2D)",
      value: hm && hm.torso_tilt_deg !== null ? `${hm.torso_tilt_deg.toFixed(0)}°` : "N/A",
      desc: "Upper-body lean/arch from vertical at contact in the camera plane.",
      color: "#4ade80",
    },
    {
      label: "Jump Timing",
      value: hm ? hm.jump_timing_label : "N/A",
      desc: "Whether ball contact happens at peak jump height or before/after the apex.",
      color: "#4ade80",
    },
    {
      label: "Block Seam",
      value: blockVal,
      desc: "Spacing between opposing blockers' hands above the net at contact.",
      color: "#fb7185",
    },
  ];
}

function drawTopLeftMechanicsHud(ctx, play) {
  const rows = getPlayMechanicsRows(play);
  if (!rows.length) return;
  const r = getVideoDisplayRect();
  const x = Math.max(10, Math.round(r.ox) + 12);
  const y = Math.max(10, Math.round(r.oy) + 12);
  const padX = 11;
  const padY = 9;
  const lineH = 18;
  const headerH = 20;

  ctx.font = "600 11px system-ui, sans-serif";
  let maxLabelW = 0;
  let maxValW = 0;
  for (const row of rows) {
    maxLabelW = Math.max(maxLabelW, ctx.measureText(row.label).width);
    maxValW = Math.max(maxValW, ctx.measureText(row.value).width);
  }
  const boxW = Math.min(els.canvas.clientWidth - x - 10, Math.max(228, Math.ceil(maxLabelW + maxValW + padX * 2 + 20)));
  const boxH = padY * 2 + headerH + rows.length * lineH;

  ctx.fillStyle = "rgba(15, 23, 42, 0.84)";
  ctx.strokeStyle = "rgba(56, 189, 248, 0.55)";
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.roundRect(x, y, boxW, boxH, 8);
  ctx.fill();
  ctx.stroke();

  ctx.font = "700 11px system-ui, sans-serif";
  ctx.fillStyle = "#38bdf8";
  ctx.fillText(`PLAY #${play.index} MECHANICS`, x + padX, y + padY + 11);

  ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x + padX, y + padY + headerH - 3);
  ctx.lineTo(x + boxW - padX, y + padY + headerH - 3);
  ctx.stroke();

  rows.forEach((row, idx) => {
    const ry = y + padY + headerH + 12 + idx * lineH;
    ctx.font = "500 11px system-ui, sans-serif";
    ctx.fillStyle = "#cbd5e1";
    ctx.textAlign = "left";
    ctx.fillText(row.label, x + padX, ry);

    ctx.font = "700 11px system-ui, sans-serif";
    ctx.fillStyle = row.color;
    ctx.textAlign = "right";
    ctx.fillText(row.value, x + boxW - padX, ry);
  });
  ctx.textAlign = "left";
}

function drawBadge(ctx, text, x, y, borderCol) {
  ctx.font = "600 12px system-ui, sans-serif";
  const padX = 7;
  const metrics = ctx.measureText(text);
  const w = metrics.width + padX * 2;
  const h = 22;
  const bx = Math.max(6, Math.min(els.canvas.clientWidth - w - 6, x));
  const by = Math.max(24, Math.min(els.canvas.clientHeight - 6, y));
  ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
  ctx.strokeStyle = borderCol;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.roundRect(bx, by - h, w, h, 5);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f8fafc";
  ctx.fillText(text, bx + padX, by - 7);
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

  if (!state.session) return;
  const cal = state.session.calibration;
  const frameIdx = currentFrameIdx();
  els.timeReadout.textContent = `${els.video.currentTime.toFixed(2)}s / f${frameIdx}`;

  if (state.overlays.netTape && cal) {
    const lp = toCanvasPt(cal.left_point);
    const rp = toCanvasPt(cal.right_point);
    ctx.strokeStyle = cal.is_manual ? "#38bdf8" : "rgba(248, 250, 252, 0.8)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(lp.x, lp.y);
    ctx.lineTo(rp.x, rp.y);
    ctx.stroke();
    for (const p of [lp, rp]) {
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (state.calibratingNet && state.calFirstPoint) {
    const p1 = toCanvasPt(state.calFirstPoint);
    ctx.fillStyle = "#facc15";
    ctx.beginPath();
    ctx.arc(p1.x, p1.y, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  const play = getActivePlay();
  if (!play) return;

  // 1. Setting Path Arc
  if (state.overlays.setArc && play.set_trajectory && play.set_trajectory.arc_points.length) {
    const traj = play.set_trajectory;
    const vis = traj.arc_points.filter((pt) => pt.frame_idx <= frameIdx);
    if (vis.length >= 2) {
      const grad = ctx.createLinearGradient(
        toCanvasPt(vis[0]).x,
        toCanvasPt(vis[0]).y,
        toCanvasPt(vis[vis.length - 1]).x,
        toCanvasPt(vis[vis.length - 1]).y
      );
      grad.addColorStop(0, "#38bdf8");
      grad.addColorStop(1, "#facc15");
      ctx.strokeStyle = grad;
      ctx.lineWidth = 3.2;
      ctx.lineCap = "round";
      ctx.beginPath();
      vis.forEach((pt, i) => {
        const cp = toCanvasPt(pt);
        if (i === 0) ctx.moveTo(cp.x, cp.y);
        else ctx.lineTo(cp.x, cp.y);
      });
      ctx.stroke();
    }
    vis.forEach((pt, idx) => {
      if (idx % 4 === 0 || idx === vis.length - 1) {
        const cp = toCanvasPt(pt);
        ctx.fillStyle = pt.is_interpolated ? "rgba(250, 204, 21, 0.55)" : "#facc15";
        ctx.beginPath();
        ctx.arc(cp.x, cp.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    if (frameIdx >= traj.peak_frame) {
      const pk = toCanvasPt(traj.peak_point);
      drawBadge(ctx, `Set Arc • ${traj.tempo_s.toFixed(2)}s`, pk.x - 45, pk.y - 12, "#38bdf8");
    }
  }

  // 2. Hitter Pose Skeleton (or all attacking candidates during Reassign mode)
  if (state.overlays.pose || state.reassigningHitter) {
    for (const track of play.tracks) {
      const isHitter = track.track_id === play.hitter_track_id;
      if (!isHitter && !state.reassigningHitter) continue;
      if (state.reassigningHitter && track.court_side !== "attacking") continue;
      const pf = track.frames.find((f) => Math.abs(f.frame_idx - frameIdx) <= 2);
      if (!pf) continue;

      const dom = play.hitter_metrics ? play.hitter_metrics.dominant_hand : "right";
      ctx.lineWidth = isHitter ? 3 : 2;
      for (const [a, b] of COCO_EDGES) {
        const ka = pf.keypoints[a];
        const kb = pf.keypoints[b];
        if (!ka || !kb || ka[2] < 0.25 || kb[2] < 0.25) continue;
        const pa = toCanvasPt({ x: ka[0], y: ka[1] });
        const pb = toCanvasPt({ x: kb[0], y: kb[1] });
        const isHitArm = isHitter && ((dom === "right" && ((a === 6 && b === 8) || (a === 8 && b === 10))) ||
          (dom === "left" && ((a === 5 && b === 7) || (a === 7 && b === 9))));
        ctx.strokeStyle = isHitArm ? "#facc15" : (isHitter ? "#4ade80" : "rgba(148, 163, 184, 0.65)");
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }
      if (state.reassigningHitter) {
        const topLeft = toCanvasPt({ x: pf.bbox[0], y: pf.bbox[1] });
        drawBadge(ctx, `Track #${track.track_id}`, topLeft.x, topLeft.y - 4, isHitter ? "#4ade80" : "#94a3b8");
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
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(cp.x, cp.y);
      ctx.lineTo(tp.x, tp.y);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(cp.x, cp.y, 10, 0, Math.PI * 2);
      ctx.stroke();

      const htText = hm.contact_height_cm !== null
        ? `${hm.contact_height_cm >= 0 ? "+" : ""}${hm.contact_height_cm.toFixed(0)}cm`
        : `${hm.contact_height_px >= 0 ? "+" : ""}${hm.contact_height_px.toFixed(0)}px`;
      drawBadge(ctx, `Contact ${htText} above tape`, cp.x + 14, cp.y - 4, "#facc15");
      if (hm.arm_extension_deg !== null) {
        drawBadge(ctx, `Arm ${hm.arm_extension_deg.toFixed(0)}° • ${hm.jump_timing_label}`, cp.x + 14, cp.y + 22, "#4ade80");
      }
    }
  }

  // 4. Opposing Blockers & Open Seam
  if (state.overlays.blockers && contactKm && Math.abs(frameIdx - contactKm.frame_idx) <= 4 && play.block_summary) {
    for (const blk of play.block_summary.blockers) {
      const tl = toCanvasPt({ x: blk.bbox[0], y: blk.bbox[1] });
      const br = toCanvasPt({ x: blk.bbox[2], y: blk.bbox[3] });
      ctx.strokeStyle = "#fb7185";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
      ctx.setLineDash([]);
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
      ctx.strokeStyle = "#fb7185";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
      drawBadge(ctx, "Block Seam Open", p1.x, p1.y - 8, "#fb7185");
    }
  }

  // 5. Always-visible Play Mechanics HUD on top-left
  drawTopLeftMechanicsHud(ctx, play);
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
    empty.textContent = "No key moments extracted for this play yet.";
    els.keyframesStrip.appendChild(empty);
    return;
  }
  for (const km of play.keyframes) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "kf-card";
    btn.addEventListener("click", () => jumpToFrame(km.frame_idx));

    const img = document.createElement("img");
    img.className = "kf-thumb";
    img.alt = km.label;
    if (km.thumbnail_rel_path && state.session) {
      img.src = `/api/sessions/${encodeURIComponent(state.session.session_id)}/thumbnails/${encodeURIComponent(km.thumbnail_rel_path)}`;
    }
    btn.appendChild(img);

    const info = document.createElement("div");
    const title = document.createElement("div");
    title.className = "kf-title";
    title.textContent = km.label;
    const meta = document.createElement("div");
    meta.className = "kf-meta";
    meta.textContent = `${km.timestamp_s.toFixed(2)}s • f${km.frame_idx}`;
    info.appendChild(title);
    info.appendChild(meta);
    btn.appendChild(info);

    els.keyframesStrip.appendChild(btn);
  }
}

function renderMetrics(play) {
  els.metricsList.replaceChildren();
  const rows = getPlayMechanicsRows(play);
  if (!rows.length) {
    const dt = document.createElement("dt");
    dt.textContent = "Status";
    const dd = document.createElement("dd");
    dd.textContent = "Select a play";
    els.metricsList.appendChild(dt);
    els.metricsList.appendChild(dd);
    return;
  }

  for (const row of rows) {
    const dt = document.createElement("dt");
    dt.textContent = row.label;
    const dd = document.createElement("dd");
    dd.textContent = row.value;
    const desc = document.createElement("div");
    desc.className = "metric-desc";
    desc.textContent = row.desc;
    els.metricsList.appendChild(dt);
    els.metricsList.appendChild(dd);
    els.metricsList.appendChild(desc);
  }
}

function renderPlaysList() {
  els.playsList.replaceChildren();
  if (!state.session || !state.session.plays.length) {
    const empty = document.createElement("div");
    empty.className = "play-sub";
    empty.textContent = state.session ? "Scanning for set-and-hit plays..." : "No plays loaded.";
    els.playsList.appendChild(empty);
    return;
  }

  for (const play of state.session.plays) {
    const item = document.createElement("div");
    item.className = "play-item" + (play.play_id === state.selectedPlayId ? " active" : "");

    const body = document.createElement("div");
    const title = document.createElement("div");
    title.className = "play-title";
    title.textContent = `Play #${play.index} • ${play.start_time_s.toFixed(1)}s`;
    const sub = document.createElement("div");
    sub.className = "play-sub";
    const tempoTxt = play.set_trajectory ? ` • ${play.set_trajectory.tempo_s.toFixed(2)}s` : "";
    sub.textContent = `${play.attack_zone}${tempoTxt}`;
    body.appendChild(title);
    body.appendChild(sub);
    body.addEventListener("click", () => selectPlay(play.play_id, true));

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "play-delete";
    delBtn.title = "Remove play";
    delBtn.textContent = "×";
    delBtn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      const updated = await apiPostJson(
        `/api/sessions/${encodeURIComponent(state.session.session_id)}/plays/${encodeURIComponent(play.play_id)}/delete`,
        {}
      );
      applySession(updated, false);
    });

    item.appendChild(body);
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
    els.statusBadge.textContent = "No Video Loaded";
    return;
  }
  if (sess.status === "ready") {
    els.statusBadge.classList.add("status-ready");
    els.statusBadge.textContent = `Ready • ${sess.plays.length} play(s)`;
  } else if (sess.status === "error") {
    els.statusBadge.classList.add("status-error");
    els.statusBadge.textContent = sess.error_message || "Analysis Error";
  } else {
    els.statusBadge.classList.add("status-busy");
    els.statusBadge.textContent = `${sess.status.toUpperCase()} (${Math.round(sess.progress_pct)}%)`;
  }
}

function applySession(sess, resetVideoSrc) {
  const prevId = state.session ? state.session.session_id : null;
  state.session = sess;
  updateStatusPill(sess);
  els.netHeightSelect.value = String(sess.calibration.net_height_m || 2.43);

  if (resetVideoSrc || prevId !== sess.session_id) {
    els.emptyState.classList.add("hidden");
    els.video.src = `/api/sessions/${encodeURIComponent(sess.session_id)}/video`;
    els.seekSlider.max = String(Math.max(1, sess.metadata.duration_s));
    state.selectedPlayId = sess.plays.length ? sess.plays[0].play_id : null;
  } else if (!state.selectedPlayId && sess.plays.length) {
    state.selectedPlayId = sess.plays[0].play_id;
  } else if (state.selectedPlayId && !sess.plays.some((p) => p.play_id === state.selectedPlayId)) {
    state.selectedPlayId = sess.plays.length ? sess.plays[0].play_id : null;
  }

  const activePlay = getActivePlay();
  renderPlaysList();
  renderKeyframes(activePlay);
  renderMetrics(activePlay);
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
  els.playPauseBtn.textContent = els.video.paused ? "Play" : "Pause";
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
  state.reassigningHitter = reassign;
  els.calibrateNetBtn.classList.toggle("active-mode", calNet);
  els.reassignHitterBtn.classList.toggle("active-mode", reassign);
  els.canvas.classList.toggle("interactive", calNet || reassign);
  els.modeHint.classList.toggle("highlight", calNet || reassign);

  if (calNet) {
    els.modeHint.textContent = "Step 1/2: Click the LEFT antenna top-tape corner on the video.";
  } else if (reassign) {
    els.modeHint.textContent = "Click any attacking player skeleton on the video to assign them as Active Hitter.";
  } else {
    els.modeHint.textContent = "Space: Play/Pause • ←/→: Step Frame • 1/2/3: Key Moments";
  }
  renderOverlay();
}

function setupEventListeners() {
  const openModal = () => {
    els.loadError.textContent = "";
    els.loadDialog.showModal();
  };
  els.openLoadBtn.addEventListener("click", openModal);
  els.emptyLoadBtn.addEventListener("click", openModal);
  els.closeLoadBtn.addEventListener("click", () => els.loadDialog.close());

  els.uploadForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    els.loadError.textContent = "";
    const file = els.videoFileInput.files[0];
    if (!file) {
      els.loadError.textContent = "Choose a video file first.";
      return;
    }
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
      return;
    }
    const sess = await res.json();
    els.loadDialog.close();
    applySession(sess, true);
    refreshSessionsDropdown();
  });

  els.localPathForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    els.loadError.textContent = "";
    const rawPath = els.localPathInput.value.trim();
    if (!rawPath) {
      els.loadError.textContent = "Enter a local file path.";
      return;
    }
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
      btn.classList.toggle("active", state.overlays[key]);
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
      const frameIdx = currentFrameIdx();
      let bestId = null;
      let bestDist = 140;
      for (const track of play.tracks) {
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
      document.querySelectorAll(".speed-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
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
    } finally {
      els.addPlayBtn.disabled = false;
    }
  });

  els.exportMp4Btn.addEventListener("click", async () => {
    const play = getActivePlay();
    if (!state.session || !play) return;
    els.exportMp4Btn.disabled = true;
    els.exportStatus.textContent = "Rendering annotated MP4...";
    try {
      const res = await apiPostJson(
        `/api/sessions/${encodeURIComponent(state.session.session_id)}/plays/${encodeURIComponent(play.play_id)}/export`,
        { freeze_contact: els.freezeContactChk.checked }
      );
      els.exportStatus.replaceChildren();
      const link = document.createElement("a");
      link.href = res.download_url;
      link.download = res.filename;
      link.textContent = `Download ${res.filename}`;
      els.exportStatus.appendChild(link);
      link.click();
    } catch (err) {
      els.exportStatus.textContent = err.message;
    } finally {
      els.exportMp4Btn.disabled = false;
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
  setupEventListeners();
  await fetchCsrf();
  await refreshSessionsDropdown();
}

init();

"use strict";

(function () {
  const COURT_W = 9.0;
  const COURT_L = 9.0;
  const PAD_X = 1.4;
  const PAD_TOP = 0.7;
  const PAD_BOT = 1.8;

  const DEFAULT_SETTER_TARGET = { x: 6.0, y: 0.8 };
  const RECEIVE_RING_RADII_M = { 3: 1.5, 2: 3.2, 1: 4.8 };
  const RECEIVE_SCORE_COLORS = {
    3: "#22c55e",
    2: "#eab308",
    1: "#ef4444",
    0: "#94a3b8",
  };

  const FORMATIONS = {
    2: [
      { label: "P1", x_m: 2.5, y_m: 6.3 },
      { label: "P2", x_m: 6.5, y_m: 6.3 },
    ],
    3: [
      { label: "P1", x_m: 1.8, y_m: 6.2 },
      { label: "P2", x_m: 4.5, y_m: 6.5 },
      { label: "P3", x_m: 7.2, y_m: 6.2 },
    ],
    4: [
      { label: "P1", x_m: 1.4, y_m: 6.0 },
      { label: "P2", x_m: 3.5, y_m: 6.5 },
      { label: "P3", x_m: 5.5, y_m: 6.5 },
      { label: "P4", x_m: 7.6, y_m: 6.0 },
    ],
  };

  const CAL_LABELS = [
    "1/4: Click Left Sideline at Net Centerline",
    "2/4: Click Right Sideline at Net Centerline",
    "3/4: Click Right Sideline at 3m Attack Line",
    "4/4: Click Left Sideline at 3m Attack Line",
  ];

  const NET_CAL_LABELS = [
    "1/3: Click Left Antenna Top-Tape Corner",
    "2/3: Click Right Antenna Top-Tape Corner",
    "3/3: Click Any Point on Left or Right Sideline (Near Court)",
  ];

  const SPLIT_CELLS = [
    ["short", "Short"],
    ["deep", "Deep"],
    ["seam", "Seam"],
    ["direct", "Direct"],
    ["stationary", "Stay <1m"],
    ["moving", "Move ≥1m"],
  ];

  const sState = {
    initialized: false,
    mode: "serve", // "serve" | "receive"
    sessions: [],
    videoSessions: [],
    session: null,
    summary: null,
    receiveSummary: null,
    activeServer: "All",
    resultMode: "in",
    selectedServeId: null,
    showDots: true,
    showHeatmap: false,
    showSeams: true,
    showCalOverlay: true,
    calibrating: false,
    calMode: "net", // "net" (3 clicks) | "four_corner" (4 clicks)
    calPoints: [],
    calHover: null,
    dragTarget: null, // { type: "passer" | "serve" | "setterTarget" | "pass", ... }
    pollTimer: null,
    pendingConfirm: null, // "clear" | "deleteSession" | null
    confirmTimer: null,
    manualUnlinkedSessionId: null,
    expandedGroup: null, // label of the By Slot / By Passer row whose vulnerability matrix is open
    defaultPasserLabel: null, // Match mode: passer_label applied to new manual receives
  };

  const sEls = {
    sessionSelect: document.getElementById("serveSessionSelect"),
    newSessionBtn: document.getElementById("newServeSessionBtn"),
    deleteSessionBtn: document.getElementById("deleteServeSessionBtn"),
    videoSelect: document.getElementById("serveVideoSelect"),
    loadVideoBtn: document.getElementById("serveLoadVideoBtn"),
    calibrateNetBtn: document.getElementById("calibrateNetCourtBtn"),
    calibrateBtn: document.getElementById("calibrateCourtBtn"),
    toggleCalOverlayBtn: document.getElementById("toggleCalOverlayBtn"),
    autoDetectBtn: document.getElementById("autoDetectServesBtn"),
    statusBadge: document.getElementById("serveStatusBadge"),
    statusText: document.getElementById("serveStatusText"),
    progress: document.getElementById("serveProgress"),
    banner: document.getElementById("serveMessageBanner"),
    videoCard: document.getElementById("serveVideoCard"),
    playerSlot: document.getElementById("servePlayerSlot"),
    calBadge: document.getElementById("serveCalBadge"),
    calHint: document.getElementById("serveCalHint"),
    cancelCalBtn: document.getElementById("cancelServeCalBtn"),
    courtTitle: document.getElementById("serveCourtTitle"),
    formationGroup: document.getElementById("serveFormationGroup"),
    layerToggles: document.getElementById("serveLayerToggles"),
    courtCanvas: document.getElementById("serveCourtCanvas"),
    selectionInfo: document.getElementById("serveSelectionInfo"),
    selectedResult: document.getElementById("selectedServeResult"),
    selectedReceiveScore: document.getElementById("selectedReceiveScore"),
    selectedPasserLabel: document.getElementById("selectedPasserLabel"),
    selectedSlotLabel: document.getElementById("selectedSlotLabel"),
    sessionModeGroup: document.getElementById("serveSessionModeGroup"),
    passerRosterSection: document.getElementById("passerRosterSection"),
    passerRosterChips: document.getElementById("passerRosterChips"),
    addPasserForm: document.getElementById("addPasserForm"),
    newPasserInput: document.getElementById("newPasserInput"),
    receiveTeamSplits: document.getElementById("receiveTeamSplits"),
    receiveGroupTitle: document.getElementById("receiveGroupTitle"),
    deselectBtn: document.getElementById("deselectServeBtn"),
    undoBtn: document.getElementById("undoServeBtn"),
    clearBtn: document.getElementById("clearServesBtn"),
    resetPassersBtn: document.getElementById("resetPassersBtn"),
    resetSetterTargetBtn: document.getElementById("resetSetterTargetBtn"),
    toggleDotsBtn: document.getElementById("toggleDotsBtn"),
    toggleHeatmapBtn: document.getElementById("toggleHeatmapBtn"),
    toggleSeamsBtn: document.getElementById("toggleSeamsBtn"),
    serverChips: document.getElementById("serverChips"),
    addServerForm: document.getElementById("addServerForm"),
    newServerInput: document.getElementById("newServerInput"),
    manualResultSection: document.getElementById("serveManualResultSection"),
    distSummarySection: document.getElementById("serveDistSummarySection"),
    areaBreakdownSection: document.getElementById("serveAreaBreakdownSection"),
    receiveSummarySection: document.getElementById("receiveSummarySection"),
    receiveScoredBadge: document.getElementById("receiveScoredBadge"),
    receiveSummaryCards: document.getElementById("receiveSummaryCards"),
    receiveScoreBars: document.getElementById("receiveScoreBars"),
    ringRadius3: document.getElementById("ringRadius3"),
    ringRadius2: document.getElementById("ringRadius2"),
    ringRadius1: document.getElementById("ringRadius1"),
    ringRadius3Val: document.getElementById("ringRadius3Val"),
    ringRadius2Val: document.getElementById("ringRadius2Val"),
    ringRadius1Val: document.getElementById("ringRadius1Val"),
    receivePasserSection: document.getElementById("receivePasserSection"),
    receivePasserTable: document.getElementById("receivePasserTable"),
    resultModeGroup: document.getElementById("resultModeGroup"),
    totalBadge: document.getElementById("serveTotalBadge"),
    summaryCards: document.getElementById("serveSummaryCards"),
    seamPairs: document.getElementById("serveSeamPairs"),
    depthBadge: document.getElementById("serveDepthBadge"),
    zoneBars: document.getElementById("serveZoneBars"),
    logTitle: document.getElementById("serveLogTitle"),
    logList: document.getElementById("serveLogList"),
  };

  function showBanner(msg, isError) {
    if (!msg) {
      sEls.banner.classList.add("hidden");
      sEls.banner.textContent = "";
      sEls.banner.classList.remove("is-error");
      return;
    }
    sEls.banner.textContent = msg;
    sEls.banner.classList.toggle("is-error", Boolean(isError));
    sEls.banner.classList.remove("hidden");
  }

  function sharedApi() {
    return window.VBallShared;
  }

  function mediaPlayer() {
    return sharedApi().mediaPlayer;
  }

  function withServerQuery(url) {
    if (sState.activeServer && sState.activeServer !== "All") {
      return `${url}?server=${encodeURIComponent(sState.activeServer)}`;
    }
    return url;
  }

  function postServe(url, body) {
    return sharedApi().apiPostJson(withServerQuery(url), body);
  }

  function getSelectedServe() {
    if (!sState.session || !sState.selectedServeId) return null;
    return sState.session.serves.find((s) => s.serve_id === sState.selectedServeId) || null;
  }

  function isMatchMode() {
    return Boolean(sState.session && sState.session.session_mode === "match");
  }

  async function postAndApply(url, body, errMsg) {
    try {
      showBanner("", false);
      applyServePayload(await postServe(url, body));
    } catch (err) {
      showBanner(err.message || errMsg, true);
      await refreshCurrentSession();
    }
  }

  function updateSessionFields(body, errMsg) {
    if (!sState.session) return Promise.resolve();
    return postAndApply(
      `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
      body,
      errMsg
    );
  }

  function updateSelectedServe(body, errMsg) {
    const sel = getSelectedServe();
    if (!sState.session || !sel) return Promise.resolve();
    return postAndApply(
      `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(sel.serve_id)}/update`,
      body,
      errMsg
    );
  }

  function getDisplayedPassers() {
    const sel = getSelectedServe();
    if (sel && Array.isArray(sel.passers_snapshot) && sel.passers_snapshot.length) {
      return sel.passers_snapshot;
    }
    return sState.session ? sState.session.default_passers || [] : FORMATIONS[3];
  }

  function getVisibleServes() {
    if (!sState.session) return [];
    if (sState.activeServer === "All") return sState.session.serves;
    return sState.session.serves.filter((s) => s.server_name === sState.activeServer);
  }

  function getSetterTarget() {
    const t = sState.session && sState.session.setter_target;
    if (t && typeof t.x === "number" && typeof t.y === "number") {
      return t;
    }
    return { x: DEFAULT_SETTER_TARGET.x, y: DEFAULT_SETTER_TARGET.y };
  }

  function getRingRadii() {
    const r = sState.session && sState.session.receive_ring_radii_m;
    if (Array.isArray(r) && r.length === 3 && r.every((v) => typeof v === "number" && Number.isFinite(v))) {
      return [r[0], r[1], r[2]];
    }
    return [RECEIVE_RING_RADII_M[3], RECEIVE_RING_RADII_M[2], RECEIVE_RING_RADII_M[1]];
  }

  function normalizeLocalRingRadii(radii, changedIdx) {
    let [r3, r2, r1] = radii.map((v) => Number(v));
    if (changedIdx === 0) {
      r3 = Math.max(0.4, Math.min(4.0, r3));
      r2 = Math.max(r3 + 0.3, Math.min(6.5, r2));
      r1 = Math.max(r2 + 0.3, Math.min(8.5, r1));
    } else if (changedIdx === 1) {
      r2 = Math.max(0.7, Math.min(6.5, r2));
      if (r3 > r2 - 0.3) r3 = Math.max(0.4, r2 - 0.3);
      r2 = Math.max(r3 + 0.3, r2);
      r1 = Math.max(r2 + 0.3, Math.min(8.5, r1));
    } else if (changedIdx === 2) {
      r1 = Math.max(1.0, Math.min(8.5, r1));
      if (r2 > r1 - 0.3) r2 = Math.max(0.7, r1 - 0.3);
      if (r3 > r2 - 0.3) r3 = Math.max(0.4, r2 - 0.3);
      r2 = Math.max(r3 + 0.3, r2);
      r1 = Math.max(r2 + 0.3, r1);
    } else {
      r3 = Math.max(0.4, Math.min(4.0, r3));
      r2 = Math.max(r3 + 0.3, Math.min(6.5, r2));
      r1 = Math.max(r2 + 0.3, Math.min(8.5, r1));
    }
    return [
      Math.round(r3 * 100) / 100,
      Math.round(r2 * 100) / 100,
      Math.round(r1 * 100) / 100,
    ];
  }

  function syncRingSliders() {
    const [r3, r2, r1] = getRingRadii();
    if (sEls.ringRadius3) sEls.ringRadius3.value = String(r3);
    if (sEls.ringRadius2) sEls.ringRadius2.value = String(r2);
    if (sEls.ringRadius1) sEls.ringRadius1.value = String(r1);
    if (sEls.ringRadius3Val) sEls.ringRadius3Val.textContent = `${r3.toFixed(1)}m`;
    if (sEls.ringRadius2Val) sEls.ringRadius2Val.textContent = `${r2.toFixed(1)}m`;
    if (sEls.ringRadius1Val) sEls.ringRadius1Val.textContent = `${r1.toFixed(1)}m`;
  }

  function computeLocalReceiveScore(passX, passY, target) {
    if (passY < 0) return 0;
    const [r3, r2, r1] = getRingRadii();
    const d = Math.hypot(passX - target.x, passY - target.y);
    if (d <= r3) return 3;
    if (d <= r2) return 2;
    if (d <= r1) return 1;
    return 0;
  }

  function refreshLocalReceiveScores() {
    if (!sState.session || !Array.isArray(sState.session.serves)) return;
    const target = getSetterTarget();
    for (const sv of sState.session.serves) {
      if (sv.result === "in" && sv.pass_x_m != null && sv.pass_y_m != null && !sv.receive_score_manual) {
        sv.receive_score = computeLocalReceiveScore(sv.pass_x_m, sv.pass_y_m, target);
      }
    }
  }

  function setServeMode(mode) {
    sState.mode = mode === "receive" ? "receive" : "serve";
    const isReceive = sState.mode === "receive";

    document.querySelectorAll("[data-serve-mode]").forEach((btn) => {
      const active = btn.getAttribute("data-serve-mode") === sState.mode;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-pressed", String(active));
    });

    if (sEls.courtTitle) {
      sEls.courtTitle.textContent = isReceive
        ? "Receive Quality (Setter Target)"
        : "Receiving Court (9m × 9m)";
    }
    if (sEls.formationGroup) sEls.formationGroup.classList.toggle("hidden", isReceive);
    if (sEls.layerToggles) sEls.layerToggles.classList.toggle("hidden", isReceive);
    if (sEls.resetPassersBtn) sEls.resetPassersBtn.classList.toggle("hidden", isReceive);
    if (sEls.resetSetterTargetBtn) sEls.resetSetterTargetBtn.classList.toggle("hidden", !isReceive);

    if (sEls.manualResultSection) sEls.manualResultSection.classList.toggle("hidden", isReceive);
    if (sEls.distSummarySection) sEls.distSummarySection.classList.toggle("hidden", isReceive);
    if (sEls.areaBreakdownSection) sEls.areaBreakdownSection.classList.toggle("hidden", isReceive);
    if (sEls.receiveSummarySection) sEls.receiveSummarySection.classList.toggle("hidden", !isReceive);
    if (sEls.receivePasserSection) sEls.receivePasserSection.classList.toggle("hidden", !isReceive);
    if (sEls.logTitle) sEls.logTitle.textContent = isReceive ? "Serve / Receive Log" : "Serve Log";

    renderCourt();
    renderSidebar();
  }

  // Solve 3x3 homography mapping court (0,0),(9,0),(9,3),(0,3) -> 4 image points (for live grid preview)
  function solveCourtToImageHomography(cal) {
    const src = [
      [0, 0],
      [9, 0],
      [9, 3],
      [0, 3],
    ];
    const dst = [
      [cal.front_left.x, cal.front_left.y],
      [cal.front_right.x, cal.front_right.y],
      [cal.attack_right.x, cal.attack_right.y],
      [cal.attack_left.x, cal.attack_left.y],
    ];
    const A = [];
    const b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i];
      const [u, v] = dst[i];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
      b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
      b.push(v);
    }
    // Gaussian elimination with partial pivoting
    const n = 8;
    const M = A.map((row, idx) => row.concat([b[idx]]));
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let r = col + 1; r < n; r++) {
        if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
      }
      if (Math.abs(M[pivot][col]) < 1e-9) return null;
      const tmp = M[col];
      M[col] = M[pivot];
      M[pivot] = tmp;
      const div = M[col][col];
      for (let c = col; c <= n; c++) M[col][c] /= div;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const factor = M[r][col];
        for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
      }
    }
    const h = M.map((row) => row[n]);
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  }

  function projectCourtPt(H, x, y) {
    const denom = H[6] * x + H[7] * y + H[8];
    if (denom <= 1e-4) return null;
    return {
      x: (H[0] * x + H[1] * y + H[2]) / denom,
      y: (H[3] * x + H[4] * y + H[5]) / denom,
    };
  }

  function renderServeOverlay(ctx, mp) {
    if (sEls.videoCard.classList.contains("hidden")) return;
    const toC = (pt) => mp.toCanvasPt(pt);

    // Draw saved calibration grid + net reference line (when overlay toggle is enabled)
    const cal = sState.session && sState.session.court_calibration;
    if (cal && cal.is_calibrated && !sState.calibrating && sState.showCalOverlay) {
      if (cal.net_left_top && cal.net_right_top) {
        const n1 = toC(cal.net_left_top);
        const n2 = toC(cal.net_right_top);
        ctx.save();
        ctx.strokeStyle = "rgba(74, 222, 128, 0.9)";
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(n1.x, n1.y);
        ctx.lineTo(n2.x, n2.y);
        ctx.stroke();
        ctx.restore();
      }
      const H = solveCourtToImageHomography(cal);
      if (H) {
        ctx.save();
        ctx.strokeStyle = "rgba(56, 189, 248, 0.72)";
        ctx.lineWidth = 1.5;
        const lines = [
          [[0, 0], [9, 0]],
          [[0, 3], [9, 3]],
          [[0, 9], [9, 9]],
          [[0, 0], [0, 9]],
          [[3, 0], [3, 9]],
          [[6, 0], [6, 9]],
          [[9, 0], [9, 9]],
        ];
        for (const [p1, p2] of lines) {
          const i1 = projectCourtPt(H, p1[0], p1[1]);
          const i2 = projectCourtPt(H, p2[0], p2[1]);
          if (!i1 || !i2) continue;
          const c1 = toC(i1);
          const c2 = toC(i2);
          ctx.beginPath();
          ctx.moveTo(c1.x, c1.y);
          ctx.lineTo(c2.x, c2.y);
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // Draw in-progress calibration clicks
    if (sState.calibrating) {
      ctx.save();
      const pts = sState.calPoints.map(toC);
      if (pts.length > 0) {
        ctx.strokeStyle = "#facc15";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        if (sState.calHover) {
          const hv = toC(sState.calHover);
          ctx.lineTo(hv.x, hv.y);
        }
        ctx.stroke();
      }
      pts.forEach((pt, idx) => {
        ctx.fillStyle = "#facc15";
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#090d16";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(idx + 1), pt.x, pt.y);
      });
      ctx.restore();
    }
  }

  function renderServeTimeline(ctx, _mp, layout) {
    const visible = getVisibleServes();
    if (!visible.length) return;
    const { pad, usableW, trackY, trackH, duration } = layout;

    for (const sv of visible) {
      if (sv.timestamp_s === null || sv.timestamp_s === undefined) continue;
      const sx = pad + Math.max(0, Math.min(1, sv.timestamp_s / duration)) * usableW;
      const isSel = sv.serve_id === sState.selectedServeId;
      let color = "#38bdf8";
      if (sv.result === "ace") color = "#facc15";
      else if (sv.result === "out" || sv.result === "net") color = "#fb7185";

      if (isSel) {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(sx - 2.5, trackY - 7, 5, trackH + 14);
      }
      ctx.fillStyle = color;
      ctx.fillRect(sx - 1.5, trackY - (isSel ? 6 : 4), 3, trackH + (isSel ? 12 : 8));
    }
  }

  function courtTransform(cw, ch) {
    const worldW = COURT_W + PAD_X * 2;
    const worldH = COURT_L + PAD_TOP + PAD_BOT;
    const scale = Math.min(cw / worldW, ch / worldH);
    const ox = (cw - worldW * scale) / 2 + PAD_X * scale;
    const oy = (ch - worldH * scale) / 2 + PAD_TOP * scale;
    return {
      scale,
      toPx: (x_m, y_m) => ({ x: ox + x_m * scale, y: oy + y_m * scale }),
      toCourt: (px, py) => ({ x_m: (px - ox) / scale, y_m: (py - oy) / scale }),
    };
  }

  function renderCourt() {
    const canvas = sEls.courtCanvas;
    if (!canvas) return;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (!cw || !ch) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const tf = courtTransform(cw, ch);
    const tl = tf.toPx(0, 0);
    const br = tf.toPx(COURT_W, COURT_L);

    // Out-of-bounds apron
    ctx.fillStyle = "#0e1014";
    ctx.fillRect(0, 0, cw, ch);

    // Court surface
    ctx.fillStyle = "#172233";
    ctx.fillRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);

    // Subtle front 3m zone tint
    const atkR = tf.toPx(COURT_W, 3.0);
    ctx.fillStyle = "rgba(56, 189, 248, 0.06)";
    ctx.fillRect(tl.x, tl.y, atkR.x - tl.x, atkR.y - tl.y);

    if (sState.mode === "receive") {
      renderReceiveCourtLayers(ctx, tf, tl, br, atkR);
      return;
    }

    // Seam corridors between adjacent passers
    const passers = getDisplayedPassers();
    if (sState.showSeams && passers.length >= 2) {
      const ordered = [...passers].sort((a, b) => a.x_m - b.x_m);
      ctx.save();
      for (let i = 0; i < ordered.length - 1; i++) {
        const pL = ordered[i];
        const pR = ordered[i + 1];
        const gap = pR.x_m - pL.x_m;
        if (gap <= 0.1) continue;
        const rBody = Math.min(0.75, 0.3 * gap);
        const x1 = pL.x_m + rBody;
        const x2 = pR.x_m - rBody;
        if (x2 <= x1) continue;
        const avgY = 0.5 * (pL.y_m + pR.y_m);
        const y1 = Math.max(0, avgY - 2.2);
        const y2 = Math.min(COURT_L, avgY + 2.2);
        const c1 = tf.toPx(x1, y1);
        const c2 = tf.toPx(x2, y2);
        ctx.fillStyle = "rgba(192, 132, 252, 0.15)";
        ctx.fillRect(c1.x, c1.y, c2.x - c1.x, c2.y - c1.y);
        ctx.strokeStyle = "rgba(192, 132, 252, 0.35)";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.strokeRect(c1.x, c1.y, c2.x - c1.x, c2.y - c1.y);
      }
      ctx.restore();
    }

    // Zone grid & labels
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.28)";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    for (const xm of [3.0, 6.0]) {
      const a = tf.toPx(xm, 0);
      const b = tf.toPx(xm, COURT_L);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.restore();

    // 3m attack line
    ctx.strokeStyle = "rgba(226, 232, 240, 0.72)";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(tl.x, atkR.y);
    ctx.lineTo(br.x, atkR.y);
    ctx.stroke();

    // Outer court boundary
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 2.2;
    ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);

    // Net bar at y=0
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(tl.x - 10, tl.y);
    ctx.lineTo(br.x + 10, tl.y);
    ctx.stroke();

    ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
    ctx.font = "600 11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("NET", (tl.x + br.x) / 2, tl.y - 10);

    const zoneCenters = [
      ["Z4", 1.5, 1.5],
      ["Z3", 4.5, 1.5],
      ["Z2", 7.5, 1.5],
      ["Z5", 1.5, 6.0],
      ["Z6", 4.5, 6.0],
      ["Z1", 7.5, 6.0],
    ];
    ctx.fillStyle = "rgba(148, 163, 184, 0.36)";
    ctx.font = "700 15px sans-serif";
    for (const [lbl, xm, ym] of zoneCenters) {
      const p = tf.toPx(xm, ym);
      ctx.fillText(lbl, p.x, p.y);
    }

    // Heatmap layer
    const visibleServes = getVisibleServes();
    if (sState.showHeatmap && visibleServes.length > 0) {
      ctx.save();
      for (const sv of visibleServes) {
        const pt = tf.toPx(sv.x_m, sv.y_m);
        const rad = 1.15 * tf.scale;
        const grad = ctx.createRadialGradient(pt.x, pt.y, 2, pt.x, pt.y, rad);
        grad.addColorStop(0, "rgba(250, 204, 21, 0.36)");
        grad.addColorStop(0.5, "rgba(56, 189, 248, 0.18)");
        grad.addColorStop(1, "rgba(56, 189, 248, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, rad, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Passer markers
    for (const p of passers) {
      const pt = tf.toPx(p.x_m, p.y_m);
      ctx.save();
      // Subtle personal body radius
      ctx.strokeStyle = "rgba(74, 222, 128, 0.25)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 0.75 * tf.scale, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = "#16a34a";
      ctx.strokeStyle = "#dcfce7";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 13, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.font = "700 11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(p.label, pt.x, pt.y);
      ctx.restore();
    }

    // Serve dots
    if (sState.showDots) {
      for (const sv of visibleServes) {
        const pt = tf.toPx(sv.x_m, sv.y_m);
        const isSel = sv.serve_id === sState.selectedServeId;
        let fill = "#38bdf8";
        if (sv.result === "ace") fill = "#facc15";
        else if (sv.result === "out" || sv.result === "net") fill = "#fb7185";

        ctx.save();
        if (sv.is_seam) {
          ctx.strokeStyle = "#c084fc";
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, isSel ? 12 : 9.5, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.fillStyle = fill;
        ctx.strokeStyle = isSel ? "#ffffff" : "#090d16";
        ctx.lineWidth = isSel ? 2.5 : 1.5;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, isSel ? 7.5 : 5.8, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  // Filled arrowhead pulled back 10px from `to` so it clears the destination marker
  function drawArrowHead(ctx, from, to, color) {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const len = Math.hypot(dx, dy);
    if (len <= 14) return;
    const ux = dx / len;
    const uy = dy / len;
    const tipX = to.x - ux * 10;
    const tipY = to.y - uy * 10;
    const headLen = 8;
    const headW = 4.5;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - ux * headLen - uy * headW, tipY - uy * headLen + ux * headW);
    ctx.lineTo(tipX - ux * headLen + uy * headW, tipY - uy * headLen - ux * headW);
    ctx.closePath();
    ctx.fill();
  }

  function renderReceiveCourtLayers(ctx, tf, tl, br, atkR) {
    const target = getSetterTarget();
    const tPx = tf.toPx(target.x, target.y);

    // Concentric filled scoring rings clipped to the 9m x 9m court
    ctx.save();
    ctx.beginPath();
    ctx.rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    ctx.clip();

    const [r3, r2, r1] = getRingRadii();
    const rings = [
      { score: "1", r: r1, fill: "rgba(239, 68, 68, 0.22)", stroke: "rgba(239, 68, 68, 0.78)", labelColor: "rgba(252, 165, 165, 0.9)", labelDist: 0.5 * (r2 + r1) },
      { score: "2", r: r2, fill: "rgba(234, 179, 8, 0.26)", stroke: "rgba(234, 179, 8, 0.84)", labelColor: "rgba(253, 224, 71, 0.92)", labelDist: 0.5 * (r3 + r2) },
      { score: "3", r: r3, fill: "rgba(34, 197, 94, 0.30)", stroke: "rgba(34, 197, 94, 0.88)", labelColor: "rgba(134, 239, 172, 0.95)", labelDist: r3 * 0.62 },
    ];

    for (const ring of rings) {
      const rPx = ring.r * tf.scale;
      // Repaint base court color inside each inner circle so semi-transparent tints stay pure
      ctx.fillStyle = "#172233";
      ctx.beginPath();
      ctx.arc(tPx.x, tPx.y, rPx, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = ring.fill;
      ctx.beginPath();
      ctx.arc(tPx.x, tPx.y, rPx, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = ring.stroke;
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }

    // Subtle ring score labels (3, 2, 1) along lower-left diagonal from setter target
    const labelAngle = Math.PI * 0.65;
    ctx.font = "700 13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const ring of rings) {
      const lx = target.x + Math.cos(labelAngle) * ring.labelDist;
      const ly = target.y + Math.sin(labelAngle) * ring.labelDist;
      const lPx = tf.toPx(lx, ly);
      ctx.fillStyle = ring.labelColor;
      ctx.fillText(ring.score, lPx.x, lPx.y);
    }

    // Subtle "0" label in unshaded court area
    const zeroPx = tf.toPx(1.1, 8.2);
    ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
    ctx.fillText("0", zeroPx.x, zeroPx.y);
    ctx.restore();

    // Subtle zone grid lines
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.24)";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    for (const xm of [3.0, 6.0]) {
      const a = tf.toPx(xm, 0);
      const b = tf.toPx(xm, COURT_L);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.restore();

    // 3m attack line
    ctx.strokeStyle = "rgba(226, 232, 240, 0.72)";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(tl.x, atkR.y);
    ctx.lineTo(br.x, atkR.y);
    ctx.stroke();

    // Outer court boundary
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 2.2;
    ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);

    // Net bar at y=0
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(tl.x - 10, tl.y);
    ctx.lineTo(br.x + 10, tl.y);
    ctx.stroke();

    ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
    ctx.font = "600 11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("NET", (tl.x + br.x) / 2, tl.y - 10);

    // Selected serve movement chain: pre-serve stance -> reception (video-tracked serves only)
    const sel = getSelectedServe();
    if (sel && sel.pre_passer_x_m != null && sel.pre_passer_y_m != null) {
      const recvX = sel.recv_foot_x_m != null ? sel.recv_foot_x_m : sel.x_m;
      const recvY = sel.recv_foot_y_m != null ? sel.recv_foot_y_m : sel.y_m;
      const moveM = sel.move_distance_m != null
        ? sel.move_distance_m
        : Math.hypot(recvX - sel.pre_passer_x_m, recvY - sel.pre_passer_y_m);
      const stancePt = tf.toPx(sel.pre_passer_x_m, sel.pre_passer_y_m);
      const recvPt = tf.toPx(recvX, recvY);
      ctx.save();
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(stancePt.x, stancePt.y);
      ctx.lineTo(recvPt.x, recvPt.y);
      ctx.stroke();
      drawArrowHead(ctx, stancePt, recvPt, "#fbbf24");

      // Hollow stance ring
      ctx.beginPath();
      ctx.arc(stancePt.x, stancePt.y, 7, 0, Math.PI * 2);
      ctx.stroke();

      ctx.font = "700 10px sans-serif";
      ctx.textAlign = "center";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(9, 13, 22, 0.9)";
      ctx.fillStyle = "#fde68a";
      const tags = [
        ["Stance", stancePt.x, stancePt.y + 10, "top"],
        [`${moveM.toFixed(1)}m`, (stancePt.x + recvPt.x) / 2, (stancePt.y + recvPt.y) / 2 - 5, "bottom"],
      ];
      for (const [txt, x, y, baseline] of tags) {
        ctx.textBaseline = baseline;
        ctx.strokeText(txt, x, y);
        ctx.fillText(txt, x, y);
      }
      ctx.restore();
    }

    // Selected serve origin + dashed arrow to pass destination
    if (sel && sel.pass_x_m != null && sel.pass_y_m != null) {
      const origPt = tf.toPx(sel.x_m, sel.y_m);
      const passPt = tf.toPx(sel.pass_x_m, sel.pass_y_m);
      ctx.save();
      // Dashed arrow line
      ctx.strokeStyle = "rgba(248, 250, 252, 0.85)";
      ctx.lineWidth = 1.8;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(origPt.x, origPt.y);
      ctx.lineTo(passPt.x, passPt.y);
      ctx.stroke();
      ctx.setLineDash([]);

      drawArrowHead(ctx, origPt, passPt, "rgba(248, 250, 252, 0.9)");

      // Subtle reception origin marker
      ctx.fillStyle = "rgba(56, 189, 248, 0.3)";
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(origPt.x, origPt.y, 5.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Draggable Setter Target marker
    ctx.save();
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#090d16";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(tPx.x, tPx.y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#090d16";
    ctx.beginPath();
    ctx.arc(tPx.x, tPx.y, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // "Setter" label above the target circle
    ctx.font = "700 11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = "rgba(9, 13, 22, 0.9)";
    ctx.strokeText("Setter", tPx.x, tPx.y - 13);
    ctx.fillStyle = "#f8fafc";
    ctx.fillText("Setter", tPx.x, tPx.y - 13);
    ctx.restore();

    // Pass destination dots for all visible serves with pass coordinates
    const visibleServes = getVisibleServes();
    for (const sv of visibleServes) {
      if (sv.pass_x_m == null || sv.pass_y_m == null) continue;
      const pt = tf.toPx(sv.pass_x_m, sv.pass_y_m);
      const isSel = sv.serve_id === sState.selectedServeId;
      const scoreKey = sv.receive_score != null ? sv.receive_score : 0;
      const fill = RECEIVE_SCORE_COLORS[scoreKey] || RECEIVE_SCORE_COLORS[0];

      ctx.save();
      if (isSel) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 11.5, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = fill;
      ctx.strokeStyle = isSel ? "#ffffff" : "#090d16";
      ctx.lineWidth = isSel ? 2.5 : 1.6;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isSel ? 7.5 : 5.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
  }

  function buildSplitMatrix(summary) {
    const wrap = document.createElement("div");
    wrap.className = "split-matrix";
    const grid = document.createElement("div");
    grid.className = "split-grid";
    const splits = summary.splits || {};
    for (const [key, label] of SPLIT_CELLS) {
      const b = splits[key] || {};
      const has = b.count > 0 && b.avg_rating != null;
      let tone = "";
      if (has) tone = b.avg_rating >= 2.2 ? " split-good" : b.avg_rating >= 1.5 ? " split-mid" : " split-bad";
      const cell = document.createElement("div");
      cell.className = "split-cell" + tone;
      const lbl = document.createElement("span");
      lbl.className = "split-label";
      lbl.textContent = label;
      const val = document.createElement("span");
      val.className = "split-val";
      val.textContent = has ? `${Number(b.avg_rating).toFixed(2)} (${b.count})` : "—";
      cell.appendChild(lbl);
      cell.appendChild(val);
      grid.appendChild(cell);
    }
    const pills = document.createElement("div");
    pills.className = "split-pills";
    const moveTxt = summary.move_count > 0 && summary.avg_move_m != null
      ? `${Number(summary.avg_move_m).toFixed(1)}m`
      : "N/A";
    const midTxt = summary.midline_pct != null ? `${summary.midline_pct}%` : "N/A";
    for (const txt of [`Avg Move: ${moveTxt}`, `Midline: ${midTxt}`]) {
      const pill = document.createElement("span");
      pill.className = "split-pill";
      pill.textContent = txt;
      pills.appendChild(pill);
    }
    wrap.appendChild(grid);
    wrap.appendChild(pills);
    return wrap;
  }

  function renderPasserRoster() {
    const roster = (sState.session && sState.session.passers_roster) || [];
    if (!roster.includes(sState.defaultPasserLabel)) sState.defaultPasserLabel = null;
    const sel = getSelectedServe();
    const activeName = sel ? sel.passer_label || null : sState.defaultPasserLabel;
    sEls.passerRosterChips.replaceChildren();
    for (const name of [null, ...roster]) {
      const active = activeName === name;
      const chip = document.createElement("span");
      chip.className = "roster-chip" + (active ? " active" : "");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "roster-chip-name";
      btn.textContent = name || "Unassigned";
      btn.setAttribute("aria-pressed", String(active));
      btn.addEventListener("click", async () => {
        if (!getSelectedServe()) {
          sState.defaultPasserLabel = name;
          renderSidebar();
          return;
        }
        await updateSelectedServe(name ? { passer_label: name } : { clear_passer_label: true }, "Unable to tag passer.");
      });
      chip.appendChild(btn);
      if (name) {
        const rm = document.createElement("button");
        rm.type = "button";
        rm.className = "roster-chip-remove";
        rm.setAttribute("aria-label", `Remove passer ${name}`);
        rm.textContent = "×";
        rm.addEventListener("click", async () => {
          await updateSessionFields({ passers_roster: roster.filter((n) => n !== name) }, "Unable to remove passer.");
        });
        chip.appendChild(rm);
      }
      sEls.passerRosterChips.appendChild(chip);
    }
  }

  function rebuildPasserOptions(current) {
    const roster = (sState.session && sState.session.passers_roster) || [];
    const names = ["", ...roster, ...getDisplayedPassers().map((p) => p.label)];
    if (current) names.push(current);
    const opts = [...new Set(names)].map((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name ? `Passer: ${name}` : "Passer: Unassigned";
      return opt;
    });
    sEls.selectedPasserLabel.replaceChildren(...opts);
  }

  function renderSidebar() {
    const sess = sState.session;
    const sum = sState.summary;
    const isReceive = sState.mode === "receive";
    const isMatch = isMatchMode();

    if (sEls.sessionModeGroup) {
      sEls.sessionModeGroup.querySelectorAll("[data-serve-session-mode]").forEach((btn) => {
        const active = btn.getAttribute("data-serve-session-mode") === (isMatch ? "match" : "drill");
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-pressed", String(active));
      });
    }
    if (sEls.passerRosterSection) {
      sEls.passerRosterSection.classList.toggle("hidden", !(isReceive && isMatch));
      renderPasserRoster();
    }

    // Server chips
    sEls.serverChips.replaceChildren();
    const servers = ["All"].concat(sess ? sess.servers || [] : []);
    if (sess && sess.serves.some((s) => s.server_name === "Unassigned") && !servers.includes("Unassigned")) {
      servers.push("Unassigned");
    }
    for (const srv of servers) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pill" + (sState.activeServer === srv ? " active" : "");
      btn.textContent = srv === "All" ? "All Servers" : srv;
      btn.addEventListener("click", async () => {
        sState.activeServer = srv;
        await refreshCurrentSession();
      });
      sEls.serverChips.appendChild(btn);
    }

    // Summary KPI cards
    sEls.totalBadge.textContent = String(sum ? sum.total : 0);
    sEls.summaryCards.replaceChildren();
    const kpis = [
      { label: "In %", value: `${sum ? sum.in_pct : 0}%`, cls: "accent-cyan" },
      { label: "Ace %", value: `${sum ? sum.ace_pct : 0}%`, cls: "accent-gold" },
      { label: "Between Players", value: `${sum ? sum.seam_pct : 0}%`, cls: "accent-purple" },
      { label: "Error %", value: `${sum ? sum.error_pct : 0}%`, cls: "accent-rose" },
    ];
    for (const k of kpis) {
      const card = document.createElement("div");
      card.className = "serve-kpi-card";
      const lbl = document.createElement("span");
      lbl.className = "serve-kpi-label";
      lbl.textContent = k.label;
      const val = document.createElement("span");
      val.className = `serve-kpi-value ${k.cls}`;
      val.textContent = k.value;
      card.appendChild(lbl);
      card.appendChild(val);
      sEls.summaryCards.appendChild(card);
    }

    // Seam pair chips
    sEls.seamPairs.replaceChildren();
    if (sum && sum.seam_by_pair) {
      for (const [pair, cnt] of Object.entries(sum.seam_by_pair)) {
        const chip = document.createElement("span");
        chip.className = "seam-pair-chip";
        chip.textContent = `${pair} seam: ${cnt}`;
        sEls.seamPairs.appendChild(chip);
      }
    }

    // Depth badge & Zone bars
    sEls.depthBadge.textContent = `Short ${sum ? sum.short_count : 0} · Deep ${sum ? sum.deep_count : 0}`;
    sEls.zoneBars.replaceChildren();
    const zones = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6"];
    for (const z of zones) {
      const cnt = sum && sum.zone_counts ? sum.zone_counts[z] || 0 : 0;
      const pct = sum && sum.zone_pcts ? sum.zone_pcts[z] || 0 : 0;
      const row = document.createElement("div");
      row.className = "zone-bar-row";
      const lbl = document.createElement("span");
      lbl.className = "zone-bar-label";
      lbl.textContent = z;
      const track = document.createElement("div");
      track.className = "zone-bar-track";
      const fill = document.createElement("div");
      fill.className = "zone-bar-fill";
      fill.style.width = `${Math.min(100, pct)}%`;
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "zone-bar-val";
      val.textContent = `${cnt} (${pct}%)`;
      row.appendChild(lbl);
      row.appendChild(track);
      row.appendChild(val);
      sEls.zoneBars.appendChild(row);
    }

    // Receive Quality Summary
    syncRingSliders();
    const rSum = sState.receiveSummary;
    if (sEls.receiveScoredBadge) {
      sEls.receiveScoredBadge.textContent = String(rSum ? rSum.scored_count : 0);
    }
    if (sEls.receiveSummaryCards) {
      sEls.receiveSummaryCards.replaceChildren();
      const avgText = rSum && rSum.scored_count > 0
        ? `${Number(rSum.avg_rating || 0).toFixed(2)} / 3.0`
        : "0.00 / 3.0";
      const rKpis = [
        { label: "Avg Rating", value: avgText, cls: "accent-cyan" },
        { label: "In-System", value: `${rSum ? rSum.in_system_pct : 0}%`, cls: "accent-green" },
        { label: "Perfect (3)", value: `${rSum ? rSum.perfect_pct : 0}%`, cls: "accent-gold" },
        { label: "Scored", value: String(rSum ? rSum.scored_count : 0), cls: "" },
      ];
      for (const k of rKpis) {
        const card = document.createElement("div");
        card.className = "serve-kpi-card";
        const lbl = document.createElement("span");
        lbl.className = "serve-kpi-label";
        lbl.textContent = k.label;
        const val = document.createElement("span");
        val.className = k.cls ? `serve-kpi-value ${k.cls}` : "serve-kpi-value";
        val.textContent = k.value;
        card.appendChild(lbl);
        card.appendChild(val);
        sEls.receiveSummaryCards.appendChild(card);
      }
    }

    if (sEls.receiveScoreBars) {
      sEls.receiveScoreBars.replaceChildren();
      const scoreRows = [
        { key: "3", label: "3 (Perfect)" },
        { key: "2", label: "2 (Good)" },
        { key: "1", label: "1 (Out-of-Sys)" },
        { key: "0", label: "0 (Bad)" },
      ];
      for (const sr of scoreRows) {
        const cnt = rSum && rSum.score_counts ? rSum.score_counts[sr.key] || 0 : 0;
        const pct = rSum && rSum.score_pcts ? rSum.score_pcts[sr.key] || 0 : 0;
        const row = document.createElement("div");
        row.className = "zone-bar-row receive-bar-row";
        const lbl = document.createElement("span");
        lbl.className = "zone-bar-label";
        lbl.textContent = sr.label;
        const track = document.createElement("div");
        track.className = "zone-bar-track";
        const fill = document.createElement("div");
        fill.className = `zone-bar-fill score-${sr.key}`;
        fill.style.width = `${Math.min(100, pct)}%`;
        track.appendChild(fill);
        const val = document.createElement("span");
        val.className = "zone-bar-val";
        val.textContent = `${cnt} (${pct}%)`;
        row.appendChild(lbl);
        row.appendChild(track);
        row.appendChild(val);
        sEls.receiveScoreBars.appendChild(row);
      }
    }

    if (sEls.receiveTeamSplits) {
      sEls.receiveTeamSplits.replaceChildren(...(rSum ? [buildSplitMatrix(rSum)] : []));
    }

    // By Slot (Drill) / By Passer (Match) table with expandable vulnerability matrix rows
    if (sEls.receiveGroupTitle) {
      sEls.receiveGroupTitle.textContent = isMatch ? "By Passer (Match)" : "By Slot (Drill)";
    }
    if (sEls.receivePasserTable) {
      sEls.receivePasserTable.replaceChildren();
      const groupList = rSum ? (isMatch ? rSum.by_passer : rSum.by_slot) : null;
      const groups = Array.isArray(groupList) ? groupList : [];
      if (!groups.length) {
        const empty = document.createElement("div");
        empty.className = "serve-selection-info";
        empty.textContent = "No scored passes yet.";
        sEls.receivePasserTable.appendChild(empty);
      } else {
        const head = document.createElement("div");
        head.className = "passer-row passer-row-head";
        const headers = [
          { text: isMatch ? "Passer" : "Slot", cls: "" },
          { text: "Cnt", cls: "passer-cell-num" },
          { text: "Avg", cls: "passer-cell-num" },
          { text: "In-Sys", cls: "passer-cell-num" },
        ];
        for (const h of headers) {
          const cell = document.createElement("span");
          if (h.cls) cell.className = h.cls;
          cell.textContent = h.text;
          head.appendChild(cell);
        }
        sEls.receivePasserTable.appendChild(head);

        for (const p of groups) {
          const label = p.passer_label || "Unassigned";
          const expanded = sState.expandedGroup === label;
          const row = document.createElement("div");
          row.className = "passer-row passer-row-toggle" + (expanded ? " expanded" : "");
          row.setAttribute("role", "button");
          row.setAttribute("tabindex", "0");
          row.setAttribute("aria-expanded", String(expanded));
          const toggle = () => {
            sState.expandedGroup = expanded ? null : label;
            renderSidebar();
          };
          row.addEventListener("click", toggle);
          row.addEventListener("keydown", (ev) => {
            if (ev.key === "Enter" || ev.key === " ") {
              ev.preventDefault();
              toggle();
            }
          });
          const cLbl = document.createElement("span");
          cLbl.className = "passer-cell-label";
          cLbl.textContent = label;
          const cCnt = document.createElement("span");
          cCnt.className = "passer-cell-num";
          cCnt.textContent = String(p.count || 0);
          const cAvg = document.createElement("span");
          cAvg.className = "passer-cell-num";
          cAvg.textContent = Number(p.avg_rating || 0).toFixed(2);
          const cInSys = document.createElement("span");
          cInSys.className = "passer-cell-num";
          cInSys.textContent = `${p.in_system_pct || 0}%`;
          row.appendChild(cLbl);
          row.appendChild(cCnt);
          row.appendChild(cAvg);
          row.appendChild(cInSys);
          sEls.receivePasserTable.appendChild(row);
          if (expanded) sEls.receivePasserTable.appendChild(buildSplitMatrix(p));
        }
      }
    }

    // Serve / Receive log list
    sEls.logList.replaceChildren();
    const visible = getVisibleServes();
    if (!visible.length) {
      const empty = document.createElement("div");
      empty.className = "serve-selection-info";
      empty.textContent = "No serves logged yet.";
      sEls.logList.appendChild(empty);
    } else {
      for (const sv of [...visible].reverse()) {
        const item = document.createElement("div");
        item.className = "serve-log-item" + (sv.serve_id === sState.selectedServeId ? " active" : "");
        item.setAttribute("role", "listitem");

        const meta = document.createElement("div");
        meta.className = "serve-log-meta";

        if (sv.receive_score !== null && sv.receive_score !== undefined) {
          const pill = document.createElement("span");
          pill.className = `receive-score-pill score-${sv.receive_score}`;
          pill.textContent = String(sv.receive_score);
          pill.title = `Receive score: ${sv.receive_score}`;
          meta.appendChild(pill);
        }

        if (sv.passer_label) {
          const pTag = document.createElement("span");
          pTag.className = "passer-tag";
          pTag.textContent = sv.passer_label;
          meta.appendChild(pTag);
        }

        const parts = [`#${sv.index}`, sv.zone, sv.result.toUpperCase()];
        if (sv.is_seam && sv.seam_pair) parts.push(`${sv.seam_pair[0]}-${sv.seam_pair[1]} Seam`);
        const tsVal = sState.mode === "receive" && sv.pass_timestamp_s != null ? sv.pass_timestamp_s : sv.timestamp_s;
        if (tsVal !== null && tsVal !== undefined) parts.push(`${tsVal.toFixed(1)}s`);
        const txt = document.createElement("span");
        txt.textContent = parts.join(" · ");
        meta.appendChild(txt);

        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "serve-log-del";
        delBtn.setAttribute("aria-label", `Delete serve ${sv.index}`);
        delBtn.textContent = "×";
        delBtn.addEventListener("click", async (ev) => {
          ev.stopPropagation();
          await deleteServeById(sv.serve_id);
        });

        item.addEventListener("click", () => selectServe(sv));
        item.appendChild(meta);
        item.appendChild(delBtn);
        sEls.logList.appendChild(item);
      }
    }

    // Selection info & footer controls
    const sel = getSelectedServe();
    if (sel) {
      sEls.deselectBtn.classList.remove("hidden");
      sEls.selectedResult.classList.toggle("hidden", isReceive);
      sEls.selectedResult.value = sel.result;
      if (sEls.selectedReceiveScore) {
        sEls.selectedReceiveScore.classList.toggle("hidden", !isReceive);
        sEls.selectedReceiveScore.value =
          sel.receive_score_manual && sel.receive_score != null ? String(sel.receive_score) : "auto";
      }
      if (sEls.selectedPasserLabel) {
        sEls.selectedPasserLabel.classList.toggle("hidden", !isReceive || !isMatch);
        rebuildPasserOptions(sel.passer_label);
        sEls.selectedPasserLabel.value = sel.passer_label || "";
      }
      if (sEls.selectedSlotLabel) {
        sEls.selectedSlotLabel.classList.toggle("hidden", !isReceive || isMatch);
        sEls.selectedSlotLabel.value = sel.slot_label_manual && sel.slot_label ? sel.slot_label : "auto";
      }
      if (isReceive) {
        const parts = [sel.receive_score != null ? `Score ${sel.receive_score}` : "Unscored"];
        const groupLabel = isMatch ? sel.passer_label : sel.slot_label;
        if (groupLabel) parts.push(groupLabel);
        if (sel.move_distance_m != null) parts.push(`Move ${Number(sel.move_distance_m).toFixed(1)}m`);
        if (sel.contact_alignment) parts.push(sel.contact_alignment === "midline" ? "Midline" : "Outside body");
        sEls.selectionInfo.textContent = `Selected #${sel.index} (${parts.join(" · ")}) — click court or drag dot to set pass target`;
      } else {
        const seamTxt = sel.is_seam && sel.seam_pair ? ` · ${sel.seam_pair[0]}-${sel.seam_pair[1]} Seam` : "";
        sEls.selectionInfo.textContent = `Selected #${sel.index} (${sel.zone}${seamTxt}) — drag dot or passers to edit`;
      }
    } else {
      sEls.deselectBtn.classList.add("hidden");
      sEls.selectedResult.classList.add("hidden");
      if (sEls.selectedReceiveScore) sEls.selectedReceiveScore.classList.add("hidden");
      if (sEls.selectedPasserLabel) sEls.selectedPasserLabel.classList.add("hidden");
      if (sEls.selectedSlotLabel) sEls.selectedSlotLabel.classList.add("hidden");
      sEls.selectionInfo.textContent = isReceive
        ? "Click court to log receive · Drag Setter target or pass dots"
        : "Click court to log serve · Drag passers to adjust default formation";
    }
  }

  function selectServe(sv) {
    sState.selectedServeId = sv ? sv.serve_id : null;
    const mp = mediaPlayer();
    const showVideo = !sEls.videoCard.classList.contains("hidden") && mp.hasVideo();
    if (sv && showVideo) {
      const seekTs =
        sState.mode === "receive" && sv.pass_timestamp_s != null
          ? sv.pass_timestamp_s
          : sv.timestamp_s;
      if (seekTs !== null && seekTs !== undefined) {
        mp.pause();
        mp.seekToTime(seekTs);
      }
    }
    renderCourt();
    renderSidebar();
    mp.requestRender();
  }

  async function ensureLinkedVideoForServe() {
    const mp = mediaPlayer();
    if (!sState.session || sState.session.video_session_id || !mp.hasVideo()) {
      return Boolean(sState.session && sState.session.video_session_id);
    }
    const vid = mp.getSessionId();
    if (!vid) return false;
    const payload = await postServe(
      `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
      { video_session_id: vid }
    );
    sState.session = payload.session;
    sState.summary = payload.summary;
    return true;
  }

  function applyServePayload(payload, opts = {}) {
    const manualClear = Boolean(opts.manualClear);
    sState.session = payload.session;
    sState.summary = payload.summary;
    sState.receiveSummary = payload.receive_summary || null;
    const sess = sState.session;
    const mp = mediaPlayer();
    const isServeActive = sharedApi().getActiveTab() === "serve";

    // Status pill & progress
    const statusMap = {
      idle: ["status-idle", "Ready"],
      scanning: ["status-running", `Scanning ${Math.round(sess.progress_pct || 0)}%`],
      ready: ["status-ready", "Ready"],
      error: ["status-error", "Scan Error"],
    };
    const [cls, label] = statusMap[sess.status] || statusMap.idle;
    sEls.statusBadge.className = `status-pill ${cls}`;
    sEls.statusText.textContent = label;
    sEls.progress.classList.toggle("hidden", sess.status !== "scanning");
    sEls.progress.value = sess.progress_pct || 0;
    if (sess.status === "error" && sess.error_message) showBanner(sess.error_message, true);

    if (sess.video_session_id) {
      sState.manualUnlinkedSessionId = null;
    } else if (manualClear) {
      sState.manualUnlinkedSessionId = sess.serve_session_id;
    }

    const isManuallyUnlinked = !sess.video_session_id && sState.manualUnlinkedSessionId === sess.serve_session_id;

    // Sync shared MediaPlayer only when Serve tab is currently active
    if (isServeActive) {
      if (sess.video_session_id) {
        const matched = sState.videoSessions.find((v) => v.session_id === sess.video_session_id);
        const isDifferent = mp.getSessionId() !== sess.video_session_id;
        mp.loadSession(
          matched || { session_id: sess.video_session_id, metadata: {} },
          { resetTime: isDifferent }
        );
        if (!matched) {
          const missingId = sess.video_session_id;
          fetch(`/api/sessions/${encodeURIComponent(missingId)}`, { credentials: "same-origin" })
            .then((res) => (res.ok ? res.json() : null))
            .then((fresh) => {
              if (!fresh || !fresh.session_id) return;
              if (!sState.videoSessions.some((v) => v.session_id === fresh.session_id)) {
                sState.videoSessions.push(fresh);
              }
              if (sharedApi().getActiveTab() === "serve" && mp.getSessionId() === fresh.session_id) {
                mp.loadSession(fresh, { resetTime: false });
              }
            })
            .catch(() => {});
        }
        sEls.videoSelect.value = sess.video_session_id;
      } else if (isManuallyUnlinked) {
        if (manualClear) mp.clear();
        sEls.videoSelect.value = "";
      } else if (mp.hasVideo()) {
        const sharedId = mp.getSessionId();
        const hasOption = sharedId && sState.videoSessions.some((v) => v.session_id === sharedId);
        sEls.videoSelect.value = hasOption ? sharedId : "";
      } else {
        sEls.videoSelect.value = "";
      }
    } else {
      sEls.videoSelect.value = sess.video_session_id || "";
    }

    const hasVideo = Boolean(sess.video_session_id || (!isManuallyUnlinked && isServeActive && mp.hasVideo()));
    if (sEls.toggleCalOverlayBtn) {
      sEls.toggleCalOverlayBtn.classList.toggle("hidden", !hasVideo);
      sEls.toggleCalOverlayBtn.classList.toggle("active", sState.showCalOverlay);
      sEls.toggleCalOverlayBtn.setAttribute("aria-pressed", String(sState.showCalOverlay));
    }
    sEls.videoCard.classList.toggle("hidden", !hasVideo);
    if (sEls.calibrateNetBtn) sEls.calibrateNetBtn.disabled = !hasVideo;
    sEls.calibrateBtn.disabled = !hasVideo;
    const isCal = Boolean(sess.court_calibration && sess.court_calibration.is_calibrated);
    const calMethod = sess.court_calibration && sess.court_calibration.method === "net" ? "Net Interpolated" : "4-Corner";
    sEls.calBadge.textContent = isCal ? `Calibrated (${calMethod})` : "Uncalibrated";
    sEls.autoDetectBtn.disabled = !hasVideo || !isCal || sess.status === "scanning";

    // Sync active formation preset button to session's default passer count
    const passerCount = Array.isArray(sess.default_passers) ? sess.default_passers.length : 0;
    document.querySelectorAll(".serve-formation-btn").forEach((b) => {
      const count = parseInt(b.getAttribute("data-formation") || "0", 10);
      b.classList.toggle("active", count === passerCount);
    });

    if (sess.status === "scanning") {
      if (!sState.pollTimer) {
        sState.pollTimer = window.setInterval(refreshCurrentSession, 1000);
      }
    } else if (sState.pollTimer) {
      window.clearInterval(sState.pollTimer);
      sState.pollTimer = null;
    }

    renderCourt();
    if (isServeActive) mp.requestRender();
    renderSidebar();
  }

  async function refreshVideoSessionsDropdown() {
    const res = await fetch("/api/sessions", { credentials: "same-origin" });
    if (!res.ok) return;
    sState.videoSessions = await res.json();
    sEls.videoSelect.replaceChildren();
    const noneOpt = document.createElement("option");
    noneOpt.value = "";
    noneOpt.textContent = "No Linked Video (Manual)";
    sEls.videoSelect.appendChild(noneOpt);
    for (const vs of sState.videoSessions) {
      const opt = document.createElement("option");
      opt.value = vs.session_id;
      opt.textContent = `${vs.filename} (${Math.round(vs.metadata.duration_s)}s)`;
      sEls.videoSelect.appendChild(opt);
    }
    if (sState.session && sState.session.video_session_id) {
      sEls.videoSelect.value = sState.session.video_session_id;
    }
  }

  async function refreshServeSessionsDropdown(selectId) {
    const res = await fetch("/api/serve-sessions", { credentials: "same-origin" });
    if (!res.ok) return;
    sState.sessions = await res.json();
    if (!sState.sessions.length) {
      const created = await sharedApi().apiPostJson("/api/serve-sessions", { name: "Practice Session" });
      sState.sessions = [created.session];
      applyServePayload(created);
    }
    sEls.sessionSelect.replaceChildren();
    for (const s of sState.sessions) {
      const opt = document.createElement("option");
      opt.value = s.serve_session_id;
      opt.textContent = `${s.name} (${s.serves.length})`;
      sEls.sessionSelect.appendChild(opt);
    }
    const targetId = selectId || (sState.session && sState.session.serve_session_id) || sState.sessions[0].serve_session_id;
    sEls.sessionSelect.value = targetId;
    await loadServeSession(targetId);
  }

  async function loadServeSession(id) {
    const q = sState.activeServer && sState.activeServer !== "All" ? `?server=${encodeURIComponent(sState.activeServer)}` : "";
    const res = await fetch(`/api/serve-sessions/${encodeURIComponent(id)}${q}`, { credentials: "same-origin" });
    if (!res.ok) return;
    applyServePayload(await res.json());
  }

  async function refreshCurrentSession() {
    if (!sState.session) return;
    await loadServeSession(sState.session.serve_session_id);
  }

  function clearPendingConfirm() {
    if (sState.confirmTimer) {
      window.clearTimeout(sState.confirmTimer);
      sState.confirmTimer = null;
    }
    sState.pendingConfirm = null;
    if (sEls.clearBtn) sEls.clearBtn.textContent = "Clear";
    if (sEls.deleteSessionBtn) sEls.deleteSessionBtn.textContent = "Delete";
  }

  function armConfirm(kind, btn, armedLabel) {
    clearPendingConfirm();
    sState.pendingConfirm = kind;
    btn.textContent = armedLabel;
    sState.confirmTimer = window.setTimeout(clearPendingConfirm, 4000);
  }

  async function deleteServeById(serveId) {
    if (!sState.session) return;
    try {
      showBanner("", false);
      const payload = await postServe(
        `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(serveId)}/delete`,
        {}
      );
      if (sState.selectedServeId === serveId) sState.selectedServeId = null;
      applyServePayload(payload);
    } catch (err) {
      showBanner(err.message || "Unable to delete serve.", true);
    }
  }

  function setCalibrationMode(active, mode) {
    sState.calibrating = active;
    if (mode) sState.calMode = mode;
    sState.calPoints = [];
    sState.calHover = null;
    sEls.calHint.classList.toggle("hidden", !active);
    sEls.cancelCalBtn.classList.toggle("hidden", !active);
    const labels = sState.calMode === "net" ? NET_CAL_LABELS : CAL_LABELS;
    sEls.calHint.textContent = active ? labels[0] : "";
    mediaPlayer().setCanvasInteractive(active);
    mediaPlayer().requestRender();
  }

  const serveAdapter = {
    showEmptyState: false,
    allowFileDrop: false,
    onRenderOverlay: (ctx, mp) => renderServeOverlay(ctx, mp),
    onRenderTimeline: (ctx, mp, layout) => renderServeTimeline(ctx, mp, layout),
    onCanvasMouseMove: (videoPt) => {
      if (!sState.calibrating) return;
      sState.calHover = videoPt;
      mediaPlayer().requestRender();
    },
    onCanvasMouseLeave: () => {
      if (!sState.calibrating || !sState.calHover) return;
      sState.calHover = null;
      mediaPlayer().requestRender();
    },
    onCanvasClick: async (imgPt) => {
      if (!sState.calibrating || !sState.session) return;
      sState.calPoints.push(imgPt);

      if (sState.calMode === "net") {
        if (sState.calPoints.length < 3) {
          sEls.calHint.textContent = NET_CAL_LABELS[sState.calPoints.length];
          mediaPlayer().requestRender();
          return;
        }
        const [left_top, right_top, sideline_point] = sState.calPoints;
        setCalibrationMode(false);
        const netHeightEl = document.getElementById("netHeightSelect");
        const netHeightM = netHeightEl ? parseFloat(netHeightEl.value) || 2.43 : 2.43;
        try {
          showBanner("", false);
          await ensureLinkedVideoForServe();
          const payload = await postServe(
            `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/calibration/from-net`,
            { left_top, right_top, sideline_point, net_height_m: netHeightM }
          );
          applyServePayload(payload);
        } catch (err) {
          showBanner(err.message || "Invalid net calibration.", true);
        }
        return;
      }

      if (sState.calPoints.length < 4) {
        sEls.calHint.textContent = CAL_LABELS[sState.calPoints.length];
        mediaPlayer().requestRender();
        return;
      }
      const [front_left, front_right, attack_right, attack_left] = sState.calPoints;
      setCalibrationMode(false);
      try {
        showBanner("", false);
        await ensureLinkedVideoForServe();
        const payload = await postServe(
          `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/calibration`,
          { front_left, front_right, attack_right, attack_left }
        );
        applyServePayload(payload);
      } catch (err) {
        showBanner(err.message || "Invalid 4-corner calibration.", true);
      }
    },
    onKeyDown: (ev) => {
      if (ev.code === "Escape" && sState.calibrating) {
        ev.preventDefault();
        setCalibrationMode(false);
        return true;
      }
      return false;
    },
  };

  function canvasPointerPos(canvas, ev) {
    const rect = canvas.getBoundingClientRect();
    return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
  }

  function setupServeListeners() {
    document.querySelectorAll("[data-serve-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.getAttribute("data-serve-mode") || "serve";
        setServeMode(mode);
      });
    });

    sEls.sessionSelect.addEventListener("change", () => {
      setCalibrationMode(false);
      sState.selectedServeId = null;
      sState.activeServer = "All";
      loadServeSession(sEls.sessionSelect.value);
    });

    sEls.newSessionBtn.addEventListener("click", async () => {
      try {
        showBanner("", false);
        setCalibrationMode(false);
        const label = `Practice ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
        const payload = await sharedApi().apiPostJson("/api/serve-sessions", { name: label });
        sState.selectedServeId = null;
        sState.activeServer = "All";
        applyServePayload(payload);
        await refreshServeSessionsDropdown(payload.session.serve_session_id);
      } catch (err) {
        showBanner(err.message || "Unable to create serve session.", true);
      }
    });

    if (sEls.deleteSessionBtn) {
      sEls.deleteSessionBtn.addEventListener("click", async () => {
        if (!sState.session) return;
        if (sState.pendingConfirm !== "deleteSession") {
          armConfirm("deleteSession", sEls.deleteSessionBtn, "Confirm Delete?");
          return;
        }
        clearPendingConfirm();
        setCalibrationMode(false);
        try {
          showBanner("", false);
          await sharedApi().apiPostJson(
            `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/delete`,
            {}
          );
          sState.session = null;
          sState.selectedServeId = null;
          sState.activeServer = "All";
          await refreshServeSessionsDropdown();
        } catch (err) {
          showBanner(err.message || "Unable to delete serve session.", true);
        }
      });
    }

    sEls.videoSelect.addEventListener("change", async () => {
      if (!sState.session) return;
      setCalibrationMode(false);
      const vid = sEls.videoSelect.value;
      const body = vid ? { video_session_id: vid } : { clear_video_session: true };
      try {
        showBanner("", false);
        const payload = await postServe(
          `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
          body
        );
        applyServePayload(payload, { manualClear: !vid });
      } catch (err) {
        showBanner(err.message || "Unable to update linked video.", true);
        await refreshCurrentSession();
      }
    });

    sEls.loadVideoBtn.addEventListener("click", () => {
      sharedApi().openLoadModalForServeTab();
    });

    if (sEls.calibrateNetBtn) {
      sEls.calibrateNetBtn.addEventListener("click", () => {
        if (!sState.session || (!sState.session.video_session_id && !mediaPlayer().hasVideo())) return;
        sState.showCalOverlay = true;
        if (sEls.toggleCalOverlayBtn) {
          sEls.toggleCalOverlayBtn.classList.add("active");
          sEls.toggleCalOverlayBtn.setAttribute("aria-pressed", "true");
        }
        setCalibrationMode(true, "net");
      });
    }

    sEls.calibrateBtn.addEventListener("click", () => {
      if (!sState.session || (!sState.session.video_session_id && !mediaPlayer().hasVideo())) return;
      sState.showCalOverlay = true;
      if (sEls.toggleCalOverlayBtn) {
        sEls.toggleCalOverlayBtn.classList.add("active");
        sEls.toggleCalOverlayBtn.setAttribute("aria-pressed", "true");
      }
      setCalibrationMode(true, "four_corner");
    });

    if (sEls.toggleCalOverlayBtn) {
      sEls.toggleCalOverlayBtn.addEventListener("click", () => {
        sState.showCalOverlay = !sState.showCalOverlay;
        sEls.toggleCalOverlayBtn.classList.toggle("active", sState.showCalOverlay);
        sEls.toggleCalOverlayBtn.setAttribute("aria-pressed", String(sState.showCalOverlay));
        mediaPlayer().requestRender();
      });
    }

    sEls.cancelCalBtn.addEventListener("click", () => setCalibrationMode(false));

    sEls.autoDetectBtn.addEventListener("click", async () => {
      if (!sState.session) return;
      try {
        showBanner("", false);
        await ensureLinkedVideoForServe();
        const srv = sState.activeServer === "All" ? "Unassigned" : sState.activeServer;
        const payload = await postServe(
          `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/auto-detect`,
          { server_name: srv }
        );
        applyServePayload(payload);
      } catch (err) {
        showBanner(err.message || "Unable to start serve auto-detection.", true);
      }
    });

    // Formation presets
    document.querySelectorAll(".serve-formation-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!sState.session) return;
        const count = parseInt(btn.getAttribute("data-formation") || "3", 10);
        const passers = FORMATIONS[count] || FORMATIONS[3];
        sState.selectedServeId = null;
        try {
          showBanner("", false);
          const payload = await postServe(
            `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
            { default_passers: passers }
          );
          applyServePayload(payload);
        } catch (err) {
          showBanner(err.message || "Unable to update passer formation.", true);
        }
      });
    });

    sEls.resetPassersBtn.addEventListener("click", async () => {
      if (!sState.session) return;
      sState.selectedServeId = null;
      try {
        showBanner("", false);
        const payload = await postServe(
          `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
          { default_passers: FORMATIONS[3] }
        );
        applyServePayload(payload);
      } catch (err) {
        showBanner(err.message || "Unable to reset passers.", true);
      }
    });

    if (sEls.resetSetterTargetBtn) {
      sEls.resetSetterTargetBtn.addEventListener("click", async () => {
        if (!sState.session) return;
        try {
          showBanner("", false);
          const payload = await postServe(
            `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
            {
              setter_target: { x: DEFAULT_SETTER_TARGET.x, y: DEFAULT_SETTER_TARGET.y },
              receive_ring_radii_m: [RECEIVE_RING_RADII_M[3], RECEIVE_RING_RADII_M[2], RECEIVE_RING_RADII_M[1]],
            }
          );
          applyServePayload(payload);
        } catch (err) {
          showBanner(err.message || "Unable to reset setter target.", true);
        }
      });
    }

    // Scoring ring radius sliders (live preview on input, persist on change)
    const ringSliders = [
      { el: sEls.ringRadius3, idx: 0 },
      { el: sEls.ringRadius2, idx: 1 },
      { el: sEls.ringRadius1, idx: 2 },
    ];
    for (const item of ringSliders) {
      if (!item.el) continue;
      item.el.addEventListener("input", () => {
        if (!sState.session) return;
        const cur = getRingRadii();
        cur[item.idx] = parseFloat(item.el.value);
        sState.session.receive_ring_radii_m = normalizeLocalRingRadii(cur, item.idx);
        syncRingSliders();
        refreshLocalReceiveScores();
        renderCourt();
      });
      item.el.addEventListener("change", async () => {
        if (!sState.session) return;
        try {
          showBanner("", false);
          const payload = await postServe(
            `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
            { receive_ring_radii_m: getRingRadii() }
          );
          applyServePayload(payload);
        } catch (err) {
          showBanner(err.message || "Unable to update scoring ring sizes.", true);
          await refreshCurrentSession();
        }
      });
    }

    sEls.selectedResult.addEventListener("change", async () => {
      const sel = getSelectedServe();
      if (!sState.session || !sel) return;
      try {
        showBanner("", false);
        const payload = await postServe(
          `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(sel.serve_id)}/update`,
          { result: sEls.selectedResult.value }
        );
        applyServePayload(payload);
      } catch (err) {
        showBanner(err.message || "Unable to update serve result.", true);
        await refreshCurrentSession();
      }
    });

    if (sEls.selectedReceiveScore) {
      sEls.selectedReceiveScore.addEventListener("change", async () => {
        const sel = getSelectedServe();
        if (!sState.session || !sel) return;
        const val = sEls.selectedReceiveScore.value;
        const body = val === "auto"
          ? { clear_receive_score: true }
          : { receive_score: parseInt(val, 10) };
        try {
          showBanner("", false);
          const payload = await postServe(
            `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(sel.serve_id)}/update`,
            body
          );
          applyServePayload(payload);
        } catch (err) {
          showBanner(err.message || "Unable to update receive score.", true);
          await refreshCurrentSession();
        }
      });
    }

    if (sEls.selectedPasserLabel) {
      sEls.selectedPasserLabel.addEventListener("change", async () => {
        const sel = getSelectedServe();
        if (!sState.session || !sel) return;
        const val = sEls.selectedPasserLabel.value;
        const body = val ? { passer_label: val } : { clear_passer_label: true };
        try {
          showBanner("", false);
          const payload = await postServe(
            `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(sel.serve_id)}/update`,
            body
          );
          applyServePayload(payload);
        } catch (err) {
          showBanner(err.message || "Unable to update passer label.", true);
          await refreshCurrentSession();
        }
      });
    }

    if (sEls.selectedSlotLabel) {
      sEls.selectedSlotLabel.addEventListener("change", async () => {
        const val = sEls.selectedSlotLabel.value;
        await updateSelectedServe(
          val === "auto" ? { clear_slot_label: true } : { slot_label: val },
          "Unable to update receive slot."
        );
      });
    }

    if (sEls.sessionModeGroup) {
      sEls.sessionModeGroup.querySelectorAll("[data-serve-session-mode]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const mode = btn.getAttribute("data-serve-session-mode");
          if (!sState.session || sState.session.session_mode === mode) return;
          sState.expandedGroup = null;
          await updateSessionFields({ session_mode: mode }, "Unable to switch Drill / Match mode.");
        });
      });
    }

    if (sEls.addPasserForm) {
      sEls.addPasserForm.addEventListener("submit", async (ev) => {
        ev.preventDefault();
        if (!sState.session) return;
        const name = sEls.newPasserInput.value.trim();
        if (!name) return;
        sEls.newPasserInput.value = "";
        await updateSessionFields(
          { passers_roster: [...(sState.session.passers_roster || []), name] },
          "Unable to add passer."
        );
      });
    }

    sEls.deselectBtn.addEventListener("click", () => selectServe(null));

    sEls.undoBtn.addEventListener("click", async () => {
      if (!sState.session || !sState.session.serves.length) return;
      const last = sState.session.serves[sState.session.serves.length - 1];
      await deleteServeById(last.serve_id);
    });

    if (sEls.clearBtn) {
      sEls.clearBtn.addEventListener("click", async () => {
        if (!sState.session || !sState.session.serves.length) return;
        if (sState.pendingConfirm !== "clear") {
          armConfirm("clear", sEls.clearBtn, "Confirm Clear?");
          return;
        }
        clearPendingConfirm();
        try {
          showBanner("", false);
          const payload = await postServe(
            `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/clear`,
            {}
          );
          sState.selectedServeId = null;
          applyServePayload(payload);
        } catch (err) {
          showBanner(err.message || "Unable to clear serves.", true);
        }
      });
    }

    sEls.toggleDotsBtn.addEventListener("click", () => {
      sState.showDots = !sState.showDots;
      sEls.toggleDotsBtn.classList.toggle("active", sState.showDots);
      sEls.toggleDotsBtn.setAttribute("aria-pressed", String(sState.showDots));
      renderCourt();
    });
    sEls.toggleHeatmapBtn.addEventListener("click", () => {
      sState.showHeatmap = !sState.showHeatmap;
      sEls.toggleHeatmapBtn.classList.toggle("active", sState.showHeatmap);
      sEls.toggleHeatmapBtn.setAttribute("aria-pressed", String(sState.showHeatmap));
      renderCourt();
    });
    sEls.toggleSeamsBtn.addEventListener("click", () => {
      sState.showSeams = !sState.showSeams;
      sEls.toggleSeamsBtn.classList.toggle("active", sState.showSeams);
      sEls.toggleSeamsBtn.setAttribute("aria-pressed", String(sState.showSeams));
      renderCourt();
    });

    // Result mode toggles
    sEls.resultModeGroup.querySelectorAll("[data-result]").forEach((btn) => {
      btn.addEventListener("click", () => {
        sState.resultMode = btn.getAttribute("data-result") || "in";
        sEls.resultModeGroup.querySelectorAll("[data-result]").forEach((b) => {
          const active = b === btn;
          b.classList.toggle("active", active);
          b.setAttribute("aria-pressed", String(active));
        });
      });
    });

    // Add server form
    sEls.addServerForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      if (!sState.session) return;
      const name = sEls.newServerInput.value.trim();
      if (!name) return;
      const nextServers = [...(sState.session.servers || []), name];
      sEls.newServerInput.value = "";
      sState.activeServer = name;
      try {
        showBanner("", false);
        const payload = await postServe(
          `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
          { servers: nextServers }
        );
        applyServePayload(payload);
      } catch (err) {
        showBanner(err.message || "Unable to add server.", true);
      }
    });

    // Interactive court pointer handlers (passer drag, serve select/drag, receive target/pass drag, click-to-log)
    sEls.courtCanvas.addEventListener("pointerdown", (ev) => {
      if (!sState.session) return;
      const pos = canvasPointerPos(sEls.courtCanvas, ev);
      const tf = courtTransform(sEls.courtCanvas.clientWidth, sEls.courtCanvas.clientHeight);

      if (sState.mode === "receive") {
        // Hit-test Setter Target circle first
        const target = getSetterTarget();
        const tPx = tf.toPx(target.x, target.y);
        if (Math.hypot(pos.x - tPx.x, pos.y - tPx.y) <= 15) {
          if (!sState.session.setter_target) {
            sState.session.setter_target = { x: target.x, y: target.y };
          }
          sState.dragTarget = { type: "setterTarget", startX: pos.x, startY: pos.y, moved: false };
          sEls.courtCanvas.setPointerCapture(ev.pointerId);
          return;
        }

        // Hit-test pass destination dots
        const visible = getVisibleServes();
        for (let i = visible.length - 1; i >= 0; i--) {
          const sv = visible[i];
          if (sv.pass_x_m == null || sv.pass_y_m == null) continue;
          const pt = tf.toPx(sv.pass_x_m, sv.pass_y_m);
          if (Math.hypot(pos.x - pt.x, pos.y - pt.y) <= 12) {
            selectServe(sv);
            sState.dragTarget = { type: "pass", serveId: sv.serve_id, startX: pos.x, startY: pos.y, moved: false };
            sEls.courtCanvas.setPointerCapture(ev.pointerId);
            return;
          }
        }
        return;
      }

      // Hit-test passers first
      const passers = getDisplayedPassers();
      for (let i = 0; i < passers.length; i++) {
        const pt = tf.toPx(passers[i].x_m, passers[i].y_m);
        if (Math.hypot(pos.x - pt.x, pos.y - pt.y) <= 16) {
          const sel = getSelectedServe();
          if (sel && (!Array.isArray(sel.passers_snapshot) || sel.passers_snapshot.length === 0)) {
            sel.passers_snapshot = passers.map((p) => ({ label: p.label, x_m: p.x_m, y_m: p.y_m }));
          }
          sState.dragTarget = { type: "passer", index: i, startX: pos.x, startY: pos.y, moved: false };
          sEls.courtCanvas.setPointerCapture(ev.pointerId);
          return;
        }
      }

      // Hit-test existing serves
      const visible = getVisibleServes();
      for (let i = visible.length - 1; i >= 0; i--) {
        const sv = visible[i];
        const pt = tf.toPx(sv.x_m, sv.y_m);
        if (Math.hypot(pos.x - pt.x, pos.y - pt.y) <= 11) {
          selectServe(sv);
          sState.dragTarget = { type: "serve", serveId: sv.serve_id, startX: pos.x, startY: pos.y, moved: false };
          sEls.courtCanvas.setPointerCapture(ev.pointerId);
          return;
        }
      }
    });

    sEls.courtCanvas.addEventListener("pointermove", (ev) => {
      if (!sState.session) return;
      const pos = canvasPointerPos(sEls.courtCanvas, ev);
      const tf = courtTransform(sEls.courtCanvas.clientWidth, sEls.courtCanvas.clientHeight);

      if (!sState.dragTarget) {
        if (sState.mode === "receive") {
          const target = getSetterTarget();
          const tPx = tf.toPx(target.x, target.y);
          const distPx = Math.hypot(pos.x - tPx.x, pos.y - tPx.y);
          sEls.courtCanvas.style.cursor = distPx <= 15 ? "grab" : "crosshair";
        } else {
          sEls.courtCanvas.style.cursor = "crosshair";
        }
        return;
      }

      if (!sState.dragTarget.moved && Math.hypot(pos.x - sState.dragTarget.startX, pos.y - sState.dragTarget.startY) <= 4) {
        return;
      }
      const { x_m, y_m } = tf.toCourt(pos.x, pos.y);
      sState.dragTarget.moved = true;

      if (sState.dragTarget.type === "setterTarget") {
        if (!sState.session.setter_target) {
          sState.session.setter_target = { x: DEFAULT_SETTER_TARGET.x, y: DEFAULT_SETTER_TARGET.y };
        }
        sState.session.setter_target.x = Math.max(0.5, Math.min(8.5, x_m));
        sState.session.setter_target.y = Math.max(0.2, Math.min(4.0, y_m));
        refreshLocalReceiveScores();
        renderCourt();
      } else if (sState.dragTarget.type === "pass") {
        const sv = sState.session.serves.find((s) => s.serve_id === sState.dragTarget.serveId);
        if (sv) {
          sv.pass_x_m = Math.max(-1.4, Math.min(10.4, x_m));
          sv.pass_y_m = Math.max(-0.5, Math.min(10.8, y_m));
          sv.receive_score = computeLocalReceiveScore(sv.pass_x_m, sv.pass_y_m, getSetterTarget());
          renderCourt();
        }
      } else if (sState.dragTarget.type === "passer") {
        const passers = getDisplayedPassers();
        const p = passers[sState.dragTarget.index];
        if (p) {
          p.x_m = Math.max(0.2, Math.min(8.8, x_m));
          p.y_m = Math.max(1.0, Math.min(8.8, y_m));
          renderCourt();
        }
      } else if (sState.dragTarget.type === "serve") {
        const sv = sState.session.serves.find((s) => s.serve_id === sState.dragTarget.serveId);
        if (sv) {
          sv.x_m = Math.max(-1.4, Math.min(10.4, x_m));
          sv.y_m = Math.max(-0.5, Math.min(10.8, y_m));
          renderCourt();
        }
      }
    });

    sEls.courtCanvas.addEventListener("pointercancel", async () => {
      if (!sState.dragTarget) return;
      const wasMoved = sState.dragTarget.moved;
      sState.dragTarget = null;
      if (wasMoved) {
        await refreshCurrentSession();
      }
    });

    sEls.courtCanvas.addEventListener("pointerup", async (ev) => {
      if (!sState.session) return;
      const pos = canvasPointerPos(sEls.courtCanvas, ev);
      const tf = courtTransform(sEls.courtCanvas.clientWidth, sEls.courtCanvas.clientHeight);

      if (sState.dragTarget) {
        const drag = sState.dragTarget;
        sState.dragTarget = null;
        if (!drag.moved) return;

        try {
          showBanner("", false);
          if (drag.type === "setterTarget") {
            const target = getSetterTarget();
            const payload = await postServe(
              `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
              { setter_target: { x: target.x, y: target.y } }
            );
            applyServePayload(payload);
          } else if (drag.type === "pass") {
            const sv = sState.session.serves.find((s) => s.serve_id === drag.serveId);
            if (sv) {
              const payload = await postServe(
                `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(sv.serve_id)}/update`,
                { pass_x_m: sv.pass_x_m, pass_y_m: sv.pass_y_m }
              );
              applyServePayload(payload);
            }
          } else if (drag.type === "passer") {
            const sel = getSelectedServe();
            if (sel) {
              const payload = await postServe(
                `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(sel.serve_id)}/update`,
                { passers_snapshot: sel.passers_snapshot }
              );
              applyServePayload(payload);
            } else {
              const payload = await postServe(
                `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
                { default_passers: sState.session.default_passers }
              );
              applyServePayload(payload);
            }
          } else if (drag.type === "serve") {
            const sv = sState.session.serves.find((s) => s.serve_id === drag.serveId);
            if (sv) {
              const payload = await postServe(
                `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(sv.serve_id)}/update`,
                { x_m: sv.x_m, y_m: sv.y_m }
              );
              applyServePayload(payload);
            }
          }
        } catch (err) {
          showBanner(err.message || "Unable to save drag position.", true);
          await refreshCurrentSession();
        }
        return;
      }

      // Empty court click
      const { x_m, y_m } = tf.toCourt(pos.x, pos.y);
      const clampedX = Math.max(-1.4, Math.min(10.4, x_m));
      const clampedY = Math.max(-0.5, Math.min(10.8, y_m));
      const serverName = sState.activeServer === "All" ? "Unassigned" : sState.activeServer;
      const showVideo = !sEls.videoCard.classList.contains("hidden") && mediaPlayer().hasVideo();
      const ts = showVideo ? mediaPlayer().getCurrentTime() : null;

      if (sState.mode === "receive") {
        const sel = getSelectedServe();
        if (sel && sel.result !== "in") {
          showBanner(`Cannot set pass target on a ${sel.result.toUpperCase()} serve — change result to In first.`, true);
          return;
        }
        try {
          showBanner("", false);
          if (sel) {
            const payload = await postServe(
              `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(sel.serve_id)}/update`,
              { pass_x_m: clampedX, pass_y_m: clampedY }
            );
            applyServePayload(payload);
          } else {
            if (showVideo && !sState.session.video_session_id) {
              await ensureLinkedVideoForServe();
            }
            const payload = await postServe(
              `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves`,
              {
                x_m: 4.5,
                y_m: 6.5,
                result: "in",
                server_name: serverName,
                timestamp_s: ts,
                pass_x_m: clampedX,
                pass_y_m: clampedY,
                ...(isMatchMode() && sState.defaultPasserLabel ? { passer_label: sState.defaultPasserLabel } : {}),
              }
            );
            applyServePayload(payload);
          }
        } catch (err) {
          showBanner(err.message || "Unable to log receive pass.", true);
        }
        return;
      }
      try {
        showBanner("", false);
        if (showVideo && !sState.session.video_session_id) {
          await ensureLinkedVideoForServe();
        }
        const payload = await postServe(
          `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves`,
          {
            x_m: clampedX,
            y_m: clampedY,
            result: sState.resultMode,
            server_name: serverName,
            timestamp_s: ts,
          }
        );
        applyServePayload(payload);
      } catch (err) {
        showBanner(err.message || "Unable to log serve.", true);
      }
    });

    window.addEventListener("resize", () => {
      renderCourt();
      if (sharedApi().getActiveTab() === "serve") {
        mediaPlayer().requestRender();
      }
    });
  }

  async function ensureInit() {
    if (!sState.initialized) {
      sState.initialized = true;
      setupServeListeners();
    }
    if (sharedApi().getActiveTab() === "serve") {
      mediaPlayer().mountTo(sEls.playerSlot, serveAdapter);
    }
    await refreshVideoSessionsDropdown();
    await refreshServeSessionsDropdown();
    renderCourt();
    if (sharedApi().getActiveTab() === "serve") {
      mediaPlayer().requestRender();
    }
  }

  window.VBallServeTab = {
    onTabActivated: () => {
      ensureInit();
    },
    onTabDeactivated: () => {
      if (sState.calibrating) {
        setCalibrationMode(false);
      }
    },
    onVideoLoaded: async (videoSession) => {
      await ensureInit();
      if (sharedApi().getActiveTab() === "serve") {
        mediaPlayer().loadSession(videoSession, { resetTime: true });
      }
      if (!sState.session) return;
      const payload = await sharedApi().apiPostJson(
        `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
        { video_session_id: videoSession.session_id }
      );
      applyServePayload(payload);
    },
  };
})();

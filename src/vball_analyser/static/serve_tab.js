"use strict";

(function () {
  const COURT_W = 9.0;
  const COURT_L = 9.0;
  const PAD_X = 1.4;
  const PAD_TOP = 0.7;
  const PAD_BOT = 1.8;

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

  const sState = {
    initialized: false,
    sessions: [],
    videoSessions: [],
    session: null,
    summary: null,
    activeServer: "All",
    resultMode: "in",
    selectedServeId: null,
    showDots: true,
    showHeatmap: false,
    showSeams: true,
    calibrating: false,
    calPoints: [],
    calHover: null,
    dragTarget: null, // { type: "passer" | "serve", index, serveId }
    pollTimer: null,
  };

  const sEls = {
    sessionSelect: document.getElementById("serveSessionSelect"),
    newSessionBtn: document.getElementById("newServeSessionBtn"),
    videoSelect: document.getElementById("serveVideoSelect"),
    loadVideoBtn: document.getElementById("serveLoadVideoBtn"),
    calibrateBtn: document.getElementById("calibrateCourtBtn"),
    autoDetectBtn: document.getElementById("autoDetectServesBtn"),
    statusBadge: document.getElementById("serveStatusBadge"),
    statusText: document.getElementById("serveStatusText"),
    progress: document.getElementById("serveProgress"),
    banner: document.getElementById("serveMessageBanner"),
    videoCard: document.getElementById("serveVideoCard"),
    calBadge: document.getElementById("serveCalBadge"),
    calHint: document.getElementById("serveCalHint"),
    cancelCalBtn: document.getElementById("cancelServeCalBtn"),
    playPauseBtn: document.getElementById("servePlayPauseBtn"),
    videoTime: document.getElementById("serveVideoTime"),
    video: document.getElementById("serveVideo"),
    videoCanvas: document.getElementById("serveVideoCanvas"),
    seekSlider: document.getElementById("serveSeekSlider"),
    courtCanvas: document.getElementById("serveCourtCanvas"),
    selectionInfo: document.getElementById("serveSelectionInfo"),
    selectedResult: document.getElementById("selectedServeResult"),
    deselectBtn: document.getElementById("deselectServeBtn"),
    undoBtn: document.getElementById("undoServeBtn"),
    resetPassersBtn: document.getElementById("resetPassersBtn"),
    toggleDotsBtn: document.getElementById("toggleDotsBtn"),
    toggleHeatmapBtn: document.getElementById("toggleHeatmapBtn"),
    toggleSeamsBtn: document.getElementById("toggleSeamsBtn"),
    serverChips: document.getElementById("serverChips"),
    addServerForm: document.getElementById("addServerForm"),
    newServerInput: document.getElementById("newServerInput"),
    resultModeGroup: document.getElementById("resultModeGroup"),
    totalBadge: document.getElementById("serveTotalBadge"),
    summaryCards: document.getElementById("serveSummaryCards"),
    seamPairs: document.getElementById("serveSeamPairs"),
    depthBadge: document.getElementById("serveDepthBadge"),
    zoneBars: document.getElementById("serveZoneBars"),
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
    if (Math.abs(denom) < 1e-6) return null;
    return {
      x: (H[0] * x + H[1] * y + H[2]) / denom,
      y: (H[3] * x + H[4] * y + H[5]) / denom,
    };
  }

  function getServeVideoRect() {
    const vw = sEls.video.videoWidth || 1056;
    const vh = sEls.video.videoHeight || 576;
    const cw = sEls.videoCanvas.clientWidth;
    const ch = sEls.videoCanvas.clientHeight;
    if (!cw || !ch) return { ox: 0, oy: 0, dw: cw, dh: ch, scale: 1 };
    const scale = Math.min(cw / vw, ch / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    return { ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw, dh, scale };
  }

  function renderVideoOverlay() {
    const canvas = sEls.videoCanvas;
    if (!canvas || sEls.videoCard.classList.contains("hidden")) return;
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

    const rect = getServeVideoRect();
    const toC = (pt) => ({ x: rect.ox + pt.x * rect.scale, y: rect.oy + pt.y * rect.scale });

    // Draw saved calibration grid
    const cal = sState.session && sState.session.court_calibration;
    if (cal && cal.is_calibrated && !sState.calibrating) {
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
    ctx.fillStyle = "#0b1324";
    ctx.fillRect(0, 0, cw, ch);

    // Court surface
    ctx.fillStyle = "#142642";
    ctx.fillRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);

    // Subtle front 3m zone tint
    const atkR = tf.toPx(COURT_W, 3.0);
    ctx.fillStyle = "rgba(56, 189, 248, 0.06)";
    ctx.fillRect(tl.x, tl.y, atkR.x - tl.x, atkR.y - tl.y);

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

  function renderSidebar() {
    const sess = sState.session;
    const sum = sState.summary;

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

    // Serve log list
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
        const parts = [`#${sv.index}`, sv.zone, sv.result.toUpperCase()];
        if (sv.is_seam && sv.seam_pair) parts.push(`${sv.seam_pair[0]}-${sv.seam_pair[1]} Seam`);
        if (sv.timestamp_s !== null && sv.timestamp_s !== undefined) parts.push(`${sv.timestamp_s.toFixed(1)}s`);
        meta.textContent = parts.join(" · ");

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

    // Selection info
    const sel = getSelectedServe();
    if (sel) {
      sEls.deselectBtn.classList.remove("hidden");
      sEls.selectedResult.classList.remove("hidden");
      sEls.selectedResult.value = sel.result;
      const seamTxt = sel.is_seam && sel.seam_pair ? ` · ${sel.seam_pair[0]}-${sel.seam_pair[1]} Seam` : "";
      sEls.selectionInfo.textContent = `Selected #${sel.index} (${sel.zone}${seamTxt}) — drag dot or passers to edit`;
    } else {
      sEls.deselectBtn.classList.add("hidden");
      sEls.selectedResult.classList.add("hidden");
      sEls.selectionInfo.textContent = "Click court to log serve · Drag passers to adjust default formation";
    }
  }

  function selectServe(sv) {
    sState.selectedServeId = sv ? sv.serve_id : null;
    if (sv && sv.timestamp_s !== null && sv.timestamp_s !== undefined && sEls.video.src) {
      sEls.video.currentTime = sv.timestamp_s;
      sEls.video.pause();
      sEls.playPauseBtn.textContent = "Play";
    }
    renderCourt();
    renderSidebar();
  }

  function applyServePayload(payload) {
    sState.session = payload.session;
    sState.summary = payload.summary;
    const sess = sState.session;

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

    // Linked video preview
    const hasVideo = Boolean(sess.video_session_id);
    sEls.videoCard.classList.toggle("hidden", !hasVideo);
    sEls.calibrateBtn.disabled = !hasVideo;
    const isCal = Boolean(sess.court_calibration && sess.court_calibration.is_calibrated);
    sEls.calBadge.textContent = isCal ? "Calibrated (4-Corner)" : "Uncalibrated";
    sEls.autoDetectBtn.disabled = !hasVideo || !isCal || sess.status === "scanning";

    if (hasVideo) {
      const targetSrc = `/api/sessions/${encodeURIComponent(sess.video_session_id)}/video`;
      if (!sEls.video.src.endsWith(targetSrc)) {
        sEls.video.src = targetSrc;
      }
      sEls.videoSelect.value = sess.video_session_id;
    } else {
      sEls.videoSelect.value = "";
    }

    if (sess.status === "scanning") {
      if (!sState.pollTimer) {
        sState.pollTimer = window.setInterval(refreshCurrentSession, 1000);
      }
    } else if (sState.pollTimer) {
      window.clearInterval(sState.pollTimer);
      sState.pollTimer = null;
    }

    renderCourt();
    renderVideoOverlay();
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

  async function deleteServeById(serveId) {
    if (!sState.session) return;
    const payload = await postServe(
      `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(serveId)}/delete`,
      {}
    );
    if (sState.selectedServeId === serveId) sState.selectedServeId = null;
    applyServePayload(payload);
  }

  function setCalibrationMode(active) {
    sState.calibrating = active;
    sState.calPoints = [];
    sState.calHover = null;
    sEls.calHint.classList.toggle("hidden", !active);
    sEls.cancelCalBtn.classList.toggle("hidden", !active);
    sEls.calHint.textContent = active ? CAL_LABELS[0] : "";
    renderVideoOverlay();
  }

  function canvasPointerPos(canvas, ev) {
    const rect = canvas.getBoundingClientRect();
    return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
  }

  function setupServeListeners() {
    sEls.sessionSelect.addEventListener("change", () => {
      sState.selectedServeId = null;
      sState.activeServer = "All";
      loadServeSession(sEls.sessionSelect.value);
    });

    sEls.newSessionBtn.addEventListener("click", async () => {
      const label = `Practice ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
      const payload = await sharedApi().apiPostJson("/api/serve-sessions", { name: label });
      sState.selectedServeId = null;
      sState.activeServer = "All";
      applyServePayload(payload);
      await refreshServeSessionsDropdown(payload.session.serve_session_id);
    });

    sEls.videoSelect.addEventListener("change", async () => {
      if (!sState.session) return;
      const vid = sEls.videoSelect.value;
      const body = vid ? { video_session_id: vid } : { clear_video_session: true };
      const payload = await postServe(
        `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
        body
      );
      applyServePayload(payload);
    });

    sEls.loadVideoBtn.addEventListener("click", () => {
      sharedApi().openLoadModalForServeTab();
    });

    sEls.calibrateBtn.addEventListener("click", () => {
      if (!sState.session || !sState.session.video_session_id) return;
      setCalibrationMode(true);
    });

    sEls.cancelCalBtn.addEventListener("click", () => setCalibrationMode(false));

    sEls.videoCanvas.addEventListener("mousemove", (ev) => {
      if (!sState.calibrating) return;
      const pt = canvasPointerPos(sEls.videoCanvas, ev);
      const r = getServeVideoRect();
      sState.calHover = { x: (pt.x - r.ox) / r.scale, y: (pt.y - r.oy) / r.scale };
      renderVideoOverlay();
    });

    sEls.videoCanvas.addEventListener("click", async (ev) => {
      if (!sState.calibrating || !sState.session) return;
      const pt = canvasPointerPos(sEls.videoCanvas, ev);
      const r = getServeVideoRect();
      const imgPt = { x: (pt.x - r.ox) / r.scale, y: (pt.y - r.oy) / r.scale };
      sState.calPoints.push(imgPt);
      if (sState.calPoints.length < 4) {
        sEls.calHint.textContent = CAL_LABELS[sState.calPoints.length];
        renderVideoOverlay();
        return;
      }
      const [front_left, front_right, attack_right, attack_left] = sState.calPoints;
      setCalibrationMode(false);
      try {
        showBanner("", false);
        const payload = await postServe(
          `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/calibration`,
          { front_left, front_right, attack_right, attack_left }
        );
        applyServePayload(payload);
      } catch (err) {
        showBanner(err.message || "Invalid 4-corner calibration.", true);
      }
    });

    sEls.autoDetectBtn.addEventListener("click", async () => {
      if (!sState.session) return;
      try {
        showBanner("", false);
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

    sEls.playPauseBtn.addEventListener("click", () => {
      if (!sEls.video.src) return;
      if (sEls.video.paused) {
        sEls.video.play();
        sEls.playPauseBtn.textContent = "Pause";
      } else {
        sEls.video.pause();
        sEls.playPauseBtn.textContent = "Play";
      }
    });

    sEls.video.addEventListener("loadedmetadata", () => {
      sEls.seekSlider.max = String(sEls.video.duration || 1);
      renderVideoOverlay();
    });
    sEls.video.addEventListener("timeupdate", () => {
      sEls.seekSlider.value = String(sEls.video.currentTime || 0);
      sEls.videoTime.textContent = `${(sEls.video.currentTime || 0).toFixed(2)}s`;
    });
    sEls.seekSlider.addEventListener("input", () => {
      sEls.video.currentTime = parseFloat(sEls.seekSlider.value) || 0;
    });

    // Formation presets
    document.querySelectorAll(".serve-formation-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!sState.session) return;
        const count = parseInt(btn.getAttribute("data-formation") || "3", 10);
        document.querySelectorAll(".serve-formation-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const passers = FORMATIONS[count] || FORMATIONS[3];
        sState.selectedServeId = null;
        const payload = await postServe(
          `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
          { default_passers: passers }
        );
        applyServePayload(payload);
      });
    });

    sEls.resetPassersBtn.addEventListener("click", async () => {
      if (!sState.session) return;
      sState.selectedServeId = null;
      const payload = await postServe(
        `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
        { default_passers: FORMATIONS[3] }
      );
      applyServePayload(payload);
    });

    sEls.selectedResult.addEventListener("change", async () => {
      const sel = getSelectedServe();
      if (!sState.session || !sel) return;
      const payload = await postServe(
        `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/serves/${encodeURIComponent(sel.serve_id)}/update`,
        { result: sEls.selectedResult.value }
      );
      applyServePayload(payload);
    });

    sEls.deselectBtn.addEventListener("click", () => selectServe(null));

    sEls.undoBtn.addEventListener("click", async () => {
      if (!sState.session || !sState.session.serves.length) return;
      const last = sState.session.serves[sState.session.serves.length - 1];
      await deleteServeById(last.serve_id);
    });

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
      const payload = await postServe(
        `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
        { servers: nextServers }
      );
      applyServePayload(payload);
    });

    // Interactive court pointer handlers (passer drag, serve select/drag, click-to-log)
    sEls.courtCanvas.addEventListener("pointerdown", (ev) => {
      if (!sState.session) return;
      const pos = canvasPointerPos(sEls.courtCanvas, ev);
      const tf = courtTransform(sEls.courtCanvas.clientWidth, sEls.courtCanvas.clientHeight);

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
      if (!sState.dragTarget || !sState.session) return;
      const pos = canvasPointerPos(sEls.courtCanvas, ev);
      if (!sState.dragTarget.moved && Math.hypot(pos.x - sState.dragTarget.startX, pos.y - sState.dragTarget.startY) <= 4) {
        return;
      }
      const tf = courtTransform(sEls.courtCanvas.clientWidth, sEls.courtCanvas.clientHeight);
      const { x_m, y_m } = tf.toCourt(pos.x, pos.y);
      sState.dragTarget.moved = true;

      if (sState.dragTarget.type === "passer") {
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

    sEls.courtCanvas.addEventListener("pointerup", async (ev) => {
      if (!sState.session) return;
      const pos = canvasPointerPos(sEls.courtCanvas, ev);
      const tf = courtTransform(sEls.courtCanvas.clientWidth, sEls.courtCanvas.clientHeight);

      if (sState.dragTarget) {
        const drag = sState.dragTarget;
        sState.dragTarget = null;
        if (!drag.moved) return;

        if (drag.type === "passer") {
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
        return;
      }

      // Empty court click -> log new manual serve
      const { x_m, y_m } = tf.toCourt(pos.x, pos.y);
      const clampedX = Math.max(-1.4, Math.min(10.4, x_m));
      const clampedY = Math.max(-0.5, Math.min(10.8, y_m));
      const serverName = sState.activeServer === "All" ? "Unassigned" : sState.activeServer;
      const ts = sState.session.video_session_id && sEls.video.src ? sEls.video.currentTime : null;
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
    });

    window.addEventListener("resize", () => {
      renderCourt();
      renderVideoOverlay();
    });
  }

  async function ensureInit() {
    if (!sState.initialized) {
      sState.initialized = true;
      setupServeListeners();
    }
    await refreshVideoSessionsDropdown();
    await refreshServeSessionsDropdown();
    renderCourt();
    renderVideoOverlay();
  }

  window.VBallServeTab = {
    onTabActivated: () => {
      ensureInit();
    },
    onVideoLoaded: async (videoSession) => {
      await ensureInit();
      await refreshVideoSessionsDropdown();
      if (!sState.session) return;
      const payload = await sharedApi().apiPostJson(
        `/api/serve-sessions/${encodeURIComponent(sState.session.serve_session_id)}/update`,
        { video_session_id: videoSession.session_id }
      );
      applyServePayload(payload);
    },
  };
})();

"use strict";

(function () {
  class MediaPlayer {
    constructor(rootEl) {
      if (!rootEl) {
        throw new Error("MediaPlayer requires a root element");
      }
      this.root = rootEl;
      this.stage = rootEl.querySelector("#videoStage");
      this.video = rootEl.querySelector("#playerVideo");
      this.canvas = rootEl.querySelector("#overlayCanvas");
      this.emptyState = rootEl.querySelector("#emptyState");
      this.stageDropOverlay = rootEl.querySelector("#stageDropOverlay");
      this.stepBack5Btn = rootEl.querySelector("#stepBack5Btn");
      this.stepBack1Btn = rootEl.querySelector("#stepBack1Btn");
      this.playPauseBtn = rootEl.querySelector("#playPauseBtn");
      this.playPauseLabel = rootEl.querySelector("#playPauseLabel");
      this.stepFwd1Btn = rootEl.querySelector("#stepFwd1Btn");
      this.stepFwd5Btn = rootEl.querySelector("#stepFwd5Btn");
      this.timelineCanvas = rootEl.querySelector("#timelineCanvas");
      this.seekSlider = rootEl.querySelector("#seekSlider");
      this.timeReadout = rootEl.querySelector("#timeReadout");
      this.speedBtns = Array.from(rootEl.querySelectorAll(".speed-btn"));

      this._session = null;
      this._sessionId = null;
      this._fps = 30;
      this._duration = 0;
      this._videoWidth = 1280;
      this._videoHeight = 720;
      this._playbackRate = 1;
      this._renderLoopToken = 0;
      this._adapter = null;
      this._resizeObserver = null;

      this._bindEvents();
      this._syncControlsDisabled();
      this._updatePlayPauseLabel();
      this._updateEmptyStateVisibility();
    }

    _bindEvents() {
      if (this.playPauseBtn) {
        this.playPauseBtn.addEventListener("click", () => this.togglePlay());
      }
      if (this.stepBack5Btn) {
        this.stepBack5Btn.addEventListener("click", () => this.stepFrames(-5));
      }
      if (this.stepBack1Btn) {
        this.stepBack1Btn.addEventListener("click", () => this.stepFrames(-1));
      }
      if (this.stepFwd1Btn) {
        this.stepFwd1Btn.addEventListener("click", () => this.stepFrames(1));
      }
      if (this.stepFwd5Btn) {
        this.stepFwd5Btn.addEventListener("click", () => this.stepFrames(5));
      }

      if (this.seekSlider) {
        this.seekSlider.addEventListener("input", () => {
          if (!this.hasVideo()) return;
          const t = parseFloat(this.seekSlider.value);
          if (Number.isFinite(t)) {
            this.video.currentTime = Math.max(0, Math.min(this.getDuration() || t, t));
          }
          this.requestRender();
        });
      }

      for (const btn of this.speedBtns) {
        btn.addEventListener("click", () => {
          const spd = parseFloat(btn.getAttribute("data-speed"));
          if (Number.isFinite(spd) && spd > 0) {
            this.setPlaybackRate(spd);
          }
        });
      }

      if (this.canvas) {
        this.canvas.addEventListener("mousemove", (ev) => {
          if (!this.hasVideo() || !this._adapter || typeof this._adapter.onCanvasMouseMove !== "function") {
            return;
          }
          this._adapter.onCanvasMouseMove(this.toVideoPt(ev.clientX, ev.clientY), ev);
        });

        this.canvas.addEventListener("mouseleave", (ev) => {
          if (!this._adapter || typeof this._adapter.onCanvasMouseLeave !== "function") {
            return;
          }
          this._adapter.onCanvasMouseLeave(ev);
        });

        this.canvas.addEventListener("click", (ev) => {
          if (!this.hasVideo() || !this._adapter || typeof this._adapter.onCanvasClick !== "function") {
            return;
          }
          this._adapter.onCanvasClick(this.toVideoPt(ev.clientX, ev.clientY), ev);
        });
      }

      if (this.video) {
        this.video.addEventListener("timeupdate", () => {
          if (this.seekSlider) {
            this.seekSlider.value = String(this.video.currentTime || 0);
          }
          this.requestRender();
        });

        this.video.addEventListener("seeked", () => this.requestRender());

        this.video.addEventListener("loadedmetadata", () => {
          if (Number.isFinite(this.video.duration) && this.video.duration > 0) {
            this._duration = this.video.duration;
            if (this.seekSlider) {
              this.seekSlider.max = String(Math.max(1, this._duration));
            }
          }
          if (this.video.videoWidth > 0) this._videoWidth = this.video.videoWidth;
          if (this.video.videoHeight > 0) this._videoHeight = this.video.videoHeight;
          this.video.defaultPlaybackRate = this._playbackRate;
          this.video.playbackRate = this._playbackRate;
          this.requestRender();
        });

        this.video.addEventListener("play", () => {
          this._updatePlayPauseLabel();
          const token = ++this._renderLoopToken;
          const tick = () => {
            if (token !== this._renderLoopToken) return;
            this.requestRender();
            if (!this.video.paused && !this.video.ended && token === this._renderLoopToken) {
              if (typeof this.video.requestVideoFrameCallback === "function") {
                this.video.requestVideoFrameCallback(tick);
              } else {
                requestAnimationFrame(tick);
              }
            }
          };
          tick();
        });

        this.video.addEventListener("pause", () => {
          this._renderLoopToken += 1;
          this._updatePlayPauseLabel();
        });
        this.video.addEventListener("ended", () => {
          this._renderLoopToken += 1;
          this._updatePlayPauseLabel();
        });
      }

      window.addEventListener("resize", () => this.requestRender());

      if (typeof ResizeObserver === "function") {
        this._resizeObserver = new ResizeObserver(() => this.requestRender());
        if (this.stage) this._resizeObserver.observe(this.stage);
        if (this.root && this.root !== this.stage) this._resizeObserver.observe(this.root);
      }

      document.addEventListener("keydown", (ev) => {
        if (typeof document.querySelector === "function" && document.querySelector("dialog[open]")) {
          return;
        }
        const tag = ev.target && ev.target.tagName ? ev.target.tagName.toUpperCase() : "";
        const isEditable = Boolean(ev.target && ev.target.isContentEditable);
        const isSeekSlider = Boolean(this.seekSlider && ev.target === this.seekSlider);
        const inFormField = !isSeekSlider && (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || isEditable);

        if (this._adapter && typeof this._adapter.onKeyDown === "function") {
          const handled = this._adapter.onKeyDown(ev, this);
          if (handled === true) return;
        }

        if (inFormField || ev.ctrlKey || ev.metaKey || ev.altKey || !this.hasVideo() || !this._isVisible()) return;

        if (ev.code === "Space") {
          if (ev.repeat || tag === "BUTTON" || tag === "A") return;
          ev.preventDefault();
          this.togglePlay();
        } else if (ev.code === "ArrowLeft") {
          ev.preventDefault();
          this.stepFrames(-1);
        } else if (ev.code === "ArrowRight") {
          ev.preventDefault();
          this.stepFrames(1);
        }
      });
    }

    mountTo(containerEl, adapter) {
      this.pause();
      this.setCanvasInteractive(false);
      this._adapter = adapter || null;

      if (containerEl && this.root.parentNode !== containerEl) {
        containerEl.appendChild(this.root);
      }

      this._updateEmptyStateVisibility();
      if (this.stageDropOverlay && !(this._adapter && this._adapter.allowFileDrop)) {
        this.stageDropOverlay.classList.add("hidden");
        if (this.stage) this.stage.classList.remove("drag-active");
      }

      this._syncControlsDisabled();
      this.requestRender();
    }

    loadSession(videoSession, opts = {}) {
      const resetTime = Boolean(opts.resetTime);
      if (!videoSession || !videoSession.session_id) {
        this.clear();
        return false;
      }

      const meta = videoSession.metadata || {};
      const nextId = videoSession.session_id;
      const changed = this._sessionId !== nextId || !this.hasVideo();

      if (changed) {
        this._fps = 30;
        this._duration = 0;
        this._videoWidth = 1280;
        this._videoHeight = 720;
      }

      this._session = videoSession;
      this._sessionId = nextId;
      if (Number.isFinite(meta.fps) && meta.fps > 0) this._fps = meta.fps;
      if (Number.isFinite(meta.duration_s) && meta.duration_s > 0) this._duration = meta.duration_s;
      if (Number.isFinite(meta.width) && meta.width > 0) this._videoWidth = meta.width;
      if (Number.isFinite(meta.height) && meta.height > 0) this._videoHeight = meta.height;

      const targetSrc = `/api/sessions/${encodeURIComponent(nextId)}/video`;

      if (changed && this.video) {
        this._renderLoopToken += 1;
        this.video.pause();
        this.video.src = targetSrc;
        this.video.defaultPlaybackRate = this._playbackRate;
        this.video.playbackRate = this._playbackRate;
        if (resetTime) {
          this.video.currentTime = 0;
        }
      } else if (resetTime && this.video) {
        this.video.currentTime = 0;
      }

      if (this.seekSlider) {
        this.seekSlider.max = String(Math.max(1, this.getDuration() || 1));
        if (changed || resetTime) this.seekSlider.value = "0";
      }

      this._updateEmptyStateVisibility();
      this._syncControlsDisabled();
      this._updatePlayPauseLabel();
      this.requestRender();
      return changed;
    }

    clear() {
      this._renderLoopToken += 1;
      this._session = null;
      this._sessionId = null;
      this._fps = 30;
      this._duration = 0;
      this._videoWidth = 1280;
      this._videoHeight = 720;
      if (this.video) {
        this.video.pause();
        this.video.src = "";
        if (typeof this.video.removeAttribute === "function") {
          this.video.removeAttribute("src");
        }
        if (typeof this.video.load === "function") {
          this.video.load();
        }
      }
      if (this.seekSlider) {
        this.seekSlider.value = "0";
        this.seekSlider.max = "1";
      }
      if (this.timeReadout) {
        this.timeReadout.textContent = "0.00s · f0";
      }
      this.setCanvasInteractive(false);
      this._updateEmptyStateVisibility();
      this._syncControlsDisabled();
      this._updatePlayPauseLabel();
      this.requestRender();
    }

    hasVideo() {
      return Boolean(this._sessionId && this.video && this.video.src);
    }

    getSessionId() {
      return this._sessionId;
    }

    getSession() {
      return this._session;
    }

    getFps() {
      return this._fps || 30;
    }

    getDuration() {
      if (Number.isFinite(this._duration) && this._duration > 0) {
        return this._duration;
      }
      if (this.video && Number.isFinite(this.video.duration) && this.video.duration > 0) {
        return this.video.duration;
      }
      return 0;
    }

    getCurrentTime() {
      return this.video ? (Number(this.video.currentTime) || 0) : 0;
    }

    getCurrentFrame() {
      return Math.max(0, Math.round(this.getCurrentTime() * this.getFps()));
    }

    play() {
      if (!this.hasVideo()) return;
      const p = this.video.play();
      if (p && typeof p.catch === "function") {
        p.catch(() => this._updatePlayPauseLabel());
      }
      this._updatePlayPauseLabel();
    }

    pause() {
      if (!this.video) return;
      this.video.pause();
      this._updatePlayPauseLabel();
    }

    togglePlay() {
      if (!this.hasVideo()) return;
      if (this.video.paused || this.video.ended) this.play();
      else this.pause();
    }

    seekToTime(seconds) {
      if (!this.hasVideo()) return;
      const dur = this.getDuration();
      const target = Math.max(0, dur > 0 ? Math.min(dur, Number(seconds) || 0) : Math.max(0, Number(seconds) || 0));
      this.video.currentTime = target;
      if (this.seekSlider) this.seekSlider.value = String(target);
      this.requestRender();
    }

    jumpToFrame(frameIdx) {
      if (!this.hasVideo()) return;
      this.pause();
      this.seekToTime((Number(frameIdx) || 0) / this.getFps());
    }

    stepFrames(delta) {
      if (!this.hasVideo()) return;
      this.pause();
      this.seekToTime(this.getCurrentTime() + (Number(delta) || 0) / this.getFps());
    }

    setPlaybackRate(rate) {
      const safe = Number(rate) || 1;
      this._playbackRate = safe;
      if (this.video) {
        this.video.defaultPlaybackRate = safe;
        this.video.playbackRate = safe;
      }
      for (const btn of this.speedBtns) {
        const btnRate = parseFloat(btn.getAttribute("data-speed"));
        const active = Math.abs(btnRate - safe) < 1e-4;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-pressed", String(active));
      }
    }

    setCanvasInteractive(enabled) {
      if (this.canvas) {
        this.canvas.classList.toggle("interactive", Boolean(enabled));
      }
    }

    getVideoDisplayRect() {
      const vw = (this.video && this.video.videoWidth) || this._videoWidth || 1280;
      const vh = (this.video && this.video.videoHeight) || this._videoHeight || 720;
      const cw = this.canvas ? this.canvas.clientWidth : 0;
      const ch = this.canvas ? this.canvas.clientHeight : 0;
      if (!vw || !vh || !cw || !ch) return { ox: 0, oy: 0, dw: cw, dh: ch, scale: 1 };
      const scale = Math.min(cw / vw, ch / vh);
      const dw = vw * scale;
      const dh = vh * scale;
      return { ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw, dh, scale };
    }

    toCanvasPt(pt) {
      const r = this.getVideoDisplayRect();
      return { x: r.ox + pt.x * r.scale, y: r.oy + pt.y * r.scale };
    }

    toVideoPt(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      const cx = clientX - rect.left;
      const cy = clientY - rect.top;
      const r = this.getVideoDisplayRect();
      return {
        x: Math.max(0, (cx - r.ox) / r.scale),
        y: Math.max(0, (cy - r.oy) / r.scale),
      };
    }

    _isVisible() {
      if (!this.root) return false;
      if (this.root.classList && this.root.classList.contains("hidden")) return false;
      if (typeof this.root.closest === "function" && this.root.closest(".hidden")) return false;
      return true;
    }

    _updateEmptyStateVisibility() {
      if (!this.emptyState) return;
      const allowEmpty = !this._adapter || this._adapter.showEmptyState !== false;
      this.emptyState.classList.toggle("hidden", this.hasVideo() || !allowEmpty);
    }

    _syncControlsDisabled() {
      const disabled = !this.hasVideo();
      if (this.stepBack5Btn) this.stepBack5Btn.disabled = disabled;
      if (this.stepBack1Btn) this.stepBack1Btn.disabled = disabled;
      if (this.playPauseBtn) this.playPauseBtn.disabled = disabled;
      if (this.stepFwd1Btn) this.stepFwd1Btn.disabled = disabled;
      if (this.stepFwd5Btn) this.stepFwd5Btn.disabled = disabled;
      if (this.seekSlider) this.seekSlider.disabled = disabled;
    }

    _updatePlayPauseLabel() {
      if (!this.playPauseBtn || !this.playPauseLabel || !this.video) return;
      const playing = !this.video.paused && !this.video.ended;
      this.playPauseBtn.classList.toggle("is-playing", playing);
      this.playPauseLabel.textContent = playing ? "Pause" : "Play";
    }

    renderTimeline() {
      const canvas = this.timelineCanvas;
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

      if (!this.hasVideo()) return;
      const duration = Math.max(0.001, this.getDuration() || 1);
      const layout = { cw, ch, pad, usableW, trackY, trackH, duration };

      // Played fill
      const progressRatio = Math.max(0, Math.min(1, this.getCurrentTime() / duration));
      if (progressRatio > 0) {
        ctx.fillStyle = "rgba(56, 189, 248, 0.78)";
        ctx.beginPath();
        ctx.roundRect(pad, trackY, Math.max(3, progressRatio * usableW), trackH, 3);
        ctx.fill();
      }

      if (this._adapter && typeof this._adapter.onRenderTimeline === "function") {
        this._adapter.onRenderTimeline(ctx, this, layout);
      }
    }

    requestRender() {
      this.renderTimeline();

      const hasVid = this.hasVideo();
      const timeS = hasVid ? this.getCurrentTime() : 0;
      const frameIdx = hasVid ? this.getCurrentFrame() : 0;
      if (hasVid) {
        if (this.timeReadout) {
          this.timeReadout.textContent = `${timeS.toFixed(2)}s · f${frameIdx}`;
        }
        if (this._adapter && typeof this._adapter.onTimeUpdate === "function") {
          this._adapter.onTimeUpdate(timeS, frameIdx);
        }
      }

      const canvas = this.canvas;
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const cw = canvas.clientWidth;
      const ch = canvas.clientHeight;
      if (cw > 0 && ch > 0) {
        if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
          canvas.width = Math.round(cw * dpr);
          canvas.height = Math.round(ch * dpr);
        }
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cw, ch);

        if (hasVid && this._adapter && typeof this._adapter.onRenderOverlay === "function") {
          this._adapter.onRenderOverlay(ctx, this, frameIdx, this.getVideoDisplayRect());
        }
      }
    }
  }

  window.VBallMediaPlayer = { MediaPlayer };
})();

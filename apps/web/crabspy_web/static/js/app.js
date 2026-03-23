/**
 * Minimal client-side hooks for annotation tools and HTMX (see rebuild plan).
 * Prefer server-rendered flows; add canvas/SVG helpers here when needed.
 */
(function () {
  "use strict";
  document.documentElement.classList.add("js-ready");

  /** HAVE_METADATA — avoid referencing HTMLMediaElement for older engines. */
  var HAVE_METADATA = 1;

  /**
   * @param {string} text
   * @returns {number | null}
   */
  function parseTimeToSeconds(text) {
    var t = String(text).trim();
    if (!t) {
      return null;
    }
    if (t.indexOf(":") === -1) {
      var n = parseFloat(t.replace(",", "."));
      return Number.isFinite(n) ? n : null;
    }
    var parts = t.split(":").map(function (p) {
      return parseFloat(String(p).trim().replace(",", "."));
    });
    for (var i = 0; i < parts.length; i++) {
      if (!Number.isFinite(parts[i])) {
        return null;
      }
    }
    if (parts.length === 2) {
      return parts[0] * 60 + parts[1];
    }
    if (parts.length === 3) {
      return parts[0] * 3600 + parts[1] * 60 + parts[2];
    }
    return null;
  }

  /**
   * @param {HTMLVideoElement} video
   * @param {number} seconds
   */
  function seekVideo(video, seconds) {
    var apply = function () {
      var d = video.duration;
      var max = Number.isFinite(d) && d > 0 ? d : Number.POSITIVE_INFINITY;
      var target = Math.min(Math.max(0, seconds), max);
      try {
        video.currentTime = target;
      } catch (e) {
        /* ignore — invalid media may throw */
      }
    };
    if (video.readyState >= HAVE_METADATA) {
      apply();
    } else {
      video.addEventListener("loadedmetadata", apply, { once: true });
    }
  }

  /**
   * @param {{ t?: number, f?: number }} params
   */
  function replaceUrlQuery(params) {
    try {
      var url = new URL(window.location.href);
      url.search = "";
      if (params.t !== undefined && params.t !== null) {
        url.searchParams.set("t", String(params.t));
      }
      if (params.f !== undefined && params.f !== null) {
        url.searchParams.set("f", String(params.f));
      }
      window.history.replaceState(null, "", url);
    } catch (e) {
      /* ignore */
    }
  }

  /**
   * @param {HTMLElement} root
   * @param {HTMLElement | null} errEl
   * @param {string} msg
   */
  function showSeekError(errEl, msg) {
    if (!errEl) {
      return;
    }
    errEl.textContent = msg;
    errEl.hidden = false;
    errEl.setAttribute("role", "alert");
  }

  function clearSeekError(errEl) {
    if (!errEl) {
      return;
    }
    errEl.textContent = "";
    errEl.hidden = true;
    errEl.removeAttribute("role");
  }

  /**
   * @param {HTMLElement} root
   */
  function initMediaVideoViewer(root) {
    var video = root.querySelector("video");
    if (!video) {
      return;
    }

    var errEl = root.querySelector(".media-seek-error");

    var fpsRaw = root.getAttribute("data-video-fps");
    var fps =
      fpsRaw && String(fpsRaw).trim() !== ""
        ? parseFloat(fpsRaw)
        : NaN;
    var hasFps = Number.isFinite(fps) && fps > 0;

    var timeInput = root.querySelector(".media-seek-time");
    var timeBtn = root.querySelector(".media-seek-time-btn");
    var frameInput = root.querySelector(".media-seek-frame");
    var frameBtn = root.querySelector(".media-seek-frame-btn");

    if (frameInput) {
      frameInput.disabled = !hasFps;
    }
    if (frameBtn) {
      frameBtn.disabled = !hasFps;
    }

    function applyFromUrl() {
      var params = new URLSearchParams(window.location.search);
      var tParam = params.get("t");
      var fParam = params.get("f");
      if (fParam != null && fParam !== "" && hasFps) {
        var fi = parseInt(fParam, 10);
        if (Number.isFinite(fi) && fi >= 0) {
          seekVideo(video, fi / fps);
        }
      } else if (tParam != null && String(tParam).trim() !== "") {
        var ts = parseTimeToSeconds(tParam);
        if (ts != null) {
          seekVideo(video, ts);
        }
      }
    }

    applyFromUrl();

    function bindTimeGo() {
      clearSeekError(errEl);
      var raw = timeInput && timeInput.value ? timeInput.value : "";
      var sec = parseTimeToSeconds(raw);
      if (sec == null) {
        showSeekError(
          errEl,
          "Enter a time in seconds (e.g. 12.5) or MM:SS / HH:MM:SS."
        );
        return;
      }
      seekVideo(video, sec);
      replaceUrlQuery({ t: sec });
    }

    function bindFrameGo() {
      clearSeekError(errEl);
      if (!hasFps || !frameInput) {
        showSeekError(
          errEl,
          "Set frame rate on the edit page to use frame seek."
        );
        return;
      }
      var fi = parseInt(frameInput.value, 10);
      if (!Number.isFinite(fi) || fi < 0) {
        showSeekError(errEl, "Enter a non-negative frame index (0 = first frame).");
        return;
      }
      seekVideo(video, fi / fps);
      replaceUrlQuery({ f: fi });
    }

    if (timeBtn) {
      timeBtn.addEventListener("click", bindTimeGo);
    }
    if (frameBtn) {
      frameBtn.addEventListener("click", bindFrameGo);
    }
    if (timeInput) {
      timeInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
          e.preventDefault();
          bindTimeGo();
        }
      });
    }
    if (frameInput) {
      frameInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
          e.preventDefault();
          bindFrameGo();
        }
      });
    }
  }

  /**
   * Picture rectangle inside a video element (letterbox excluded).
   * @returns {{ x0: number, y0: number, width: number, height: number }}
   */
  function getVideoPictureRect(video) {
    var rect = video.getBoundingClientRect();
    var vw = video.videoWidth;
    var vh = video.videoHeight;
    var elW = rect.width;
    var elH = rect.height;
    if (!vw || !vh) {
      return { x0: 0, y0: 0, width: elW, height: elH };
    }
    var scale = Math.min(elW / vw, elH / vh);
    var dispW = vw * scale;
    var dispH = vh * scale;
    var x0 = (elW - dispW) / 2;
    var y0 = (elH - dispH) / 2;
    return { x0: x0, y0: y0, width: dispW, height: dispH };
  }

  /**
   * Picture rectangle for a letterboxed img element (contain-style fit).
   * @returns {{ x0: number, y0: number, width: number, height: number }}
   */
  function getImgPictureRect(img) {
    var rect = img.getBoundingClientRect();
    var nw = img.naturalWidth;
    var nh = img.naturalHeight;
    var elW = rect.width;
    var elH = rect.height;
    if (!nw || !nh) {
      return { x0: 0, y0: 0, width: elW, height: elH };
    }
    var scale = Math.min(elW / nw, elH / nh);
    var dispW = nw * scale;
    var dispH = nh * scale;
    var x0 = (elW - dispW) / 2;
    var y0 = (elH - dispH) / 2;
    return { x0: x0, y0: y0, width: dispW, height: dispH };
  }

  /**
   * @param {HTMLVideoElement | HTMLImageElement} el
   */
  function getMediaPictureRect(el) {
    if (el.tagName === "VIDEO") {
      return getVideoPictureRect(el);
    }
    if (el.tagName === "IMG") {
      return getImgPictureRect(el);
    }
    var r = el.getBoundingClientRect();
    return { x0: 0, y0: 0, width: r.width, height: r.height };
  }

  /**
   * Normalized coords (0–1) within the visible picture (letterbox excluded).
   * @returns {{ x: number, y: number } | null}
   */
  function normalizedMediaClick(el, clientX, clientY) {
    var rect = el.getBoundingClientRect();
    var relX = clientX - rect.left;
    var relY = clientY - rect.top;
    var pic = getMediaPictureRect(el);
    if (relX < pic.x0 || relX > pic.x0 + pic.width || relY < pic.y0 || relY > pic.y0 + pic.height) {
      return null;
    }
    return { x: (relX - pic.x0) / pic.width, y: (relY - pic.y0) / pic.height };
  }

  /**
   * Polyline stroke in normalized SVG (0–1 viewBox) + round vertex markers as HTML (px),
   * so vertices stay circular and the stroke stays visible for any vertex count.
   * @param {HTMLElement} wrap
   * @param {Array<{ x_norm: number, y_norm: number }>} arr
   * @param {boolean} isDraft
   */
  function setPolylineWrapContents(wrap, arr, isDraft) {
    while (wrap.firstChild) {
      wrap.removeChild(wrap.firstChild);
    }
    if (!arr || arr.length === 0) {
      return;
    }
    if (arr.length === 1) {
      var only = document.createElement("div");
      only.className =
        "annotation-polyline-vertex" + (isDraft ? " annotation-polyline-vertex-draft" : "");
      only.style.left = arr[0].x_norm * 100 + "%";
      only.style.top = arr[0].y_norm * 100 + "%";
      wrap.appendChild(only);
      return;
    }
    var svgNS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 1 1");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("class", "annotation-polyline-svg" + (isDraft ? " is-draft" : ""));
    var pl = document.createElementNS(svgNS, "polyline");
    pl.setAttribute("fill", "none");
    var pair = [];
    for (var i = 0; i < arr.length; i++) {
      pair.push(arr[i].x_norm + " " + arr[i].y_norm);
    }
    pl.setAttribute("points", pair.join(" "));
    svg.appendChild(pl);
    wrap.appendChild(svg);
    for (var j = 0; j < arr.length; j++) {
      var vtx = document.createElement("div");
      vtx.className =
        "annotation-polyline-vertex" + (isDraft ? " annotation-polyline-vertex-draft" : "");
      vtx.style.left = arr[j].x_norm * 100 + "%";
      vtx.style.top = arr[j].y_norm * 100 + "%";
      wrap.appendChild(vtx);
    }
  }

  function positionPictureLayer(wrap, media) {
    var pic = getMediaPictureRect(media);
    wrap.style.position = "absolute";
    wrap.style.left = pic.x0 + "px";
    wrap.style.top = pic.y0 + "px";
    wrap.style.width = pic.width + "px";
    wrap.style.height = pic.height + "px";
    wrap.style.pointerEvents = "none";
  }

  function renderPolylineWrapFromData(wrap, media) {
    var raw = wrap.getAttribute("data-points-json");
    if (!raw) {
      return;
    }
    var arr;
    try {
      arr = JSON.parse(raw);
    } catch (e) {
      return;
    }
    if (!arr || arr.length < 2) {
      return;
    }
    positionPictureLayer(wrap, media);
    setPolylineWrapContents(wrap, arr, false);
  }

  function renderPolylineDraftEl(draftEl, media, draftPoints) {
    if (!draftEl) {
      return;
    }
    var arr = draftPoints.map(function (p) {
      return { x_norm: p.x, y_norm: p.y };
    });
    positionPictureLayer(draftEl, media);
    if (arr.length === 0) {
      while (draftEl.firstChild) {
        draftEl.removeChild(draftEl.firstChild);
      }
      return;
    }
    setPolylineWrapContents(draftEl, arr, true);
  }

  function placeDot(dot, media) {
    var xNorm = parseFloat(dot.getAttribute("data-x-norm") || "");
    var yNorm = parseFloat(dot.getAttribute("data-y-norm") || "");
    if (!Number.isFinite(xNorm) || !Number.isFinite(yNorm)) {
      return;
    }
    var pic = getMediaPictureRect(media);
    dot.style.left = pic.x0 + xNorm * pic.width + "px";
    dot.style.top = pic.y0 + yNorm * pic.height + "px";
  }

  function renderDots(overlay, media) {
    var dots = overlay.querySelectorAll(".annotation-dot");
    for (var i = 0; i < dots.length; i++) {
      placeDot(dots[i], media);
    }
  }

  function annotationAddDot(overlay, media, data) {
    var p0 = data.points && data.points[0];
    if (!p0) {
      return;
    }
    var dot = document.createElement("div");
    dot.className = "annotation-dot";
    dot.setAttribute("data-annotation-id", data.id);
    dot.setAttribute("data-x-norm", String(p0.x_norm));
    dot.setAttribute("data-y-norm", String(p0.y_norm));
    overlay.appendChild(dot);
    placeDot(dot, media);
  }

  function annotationAddPolylineWrap(overlay, media, data) {
    if (!data.points || data.points.length < 2) {
      return;
    }
    var json = JSON.stringify(
      data.points.map(function (p) {
        return { x_norm: p.x_norm, y_norm: p.y_norm };
      })
    );
    var w = document.createElement("div");
    w.className = "annotation-polyline-wrap";
    w.setAttribute("data-annotation-id", data.id);
    w.setAttribute("data-points-json", json);
    overlay.appendChild(w);
    renderPolylineWrapFromData(w, media);
  }

  function removeAnnotationVisual(overlay, aid) {
    var c = overlay.querySelector('[data-annotation-id="' + aid + '"]');
    if (c && c.parentNode) {
      c.parentNode.removeChild(c);
    }
  }

  /**
   * @param {HTMLVideoElement | HTMLImageElement} media
   * @returns {{ ref_width_px?: number, ref_height_px?: number }}
   */
  function getRefDimensions(media) {
    if (media.tagName === "VIDEO") {
      var vw = media.videoWidth;
      var vh = media.videoHeight;
      if (vw > 0 && vh > 0) {
        return { ref_width_px: vw, ref_height_px: vh };
      }
    } else if (media.tagName === "IMG") {
      var nw = media.naturalWidth;
      var nh = media.naturalHeight;
      if (nw > 0 && nh > 0) {
        return { ref_width_px: nw, ref_height_px: nh };
      }
    }
    return {};
  }

  /**
   * Optional label + intrinsic picture dimensions (video frame / image pixel size).
   * @param {HTMLElement} host
   * @param {HTMLVideoElement | HTMLImageElement} media
   * @param {Record<string, unknown>} payload
   */
  function applyAnnotationPayloadExtras(host, media, payload) {
    var r = getRefDimensions(media);
    if (r.ref_width_px) {
      payload.ref_width_px = r.ref_width_px;
      payload.ref_height_px = r.ref_height_px;
    }
    var lab = host.querySelector("[data-annotation-label]");
    if (lab && lab.value) {
      var t = String(lab.value).trim();
      if (t) {
        payload.label = t;
      }
    }
  }

  function annotationFormatRow(data) {
    var parts = [data.kind || "point"];
    if (data.label) {
      parts.push(String(data.label));
    }
    if (data.time_seconds !== null && data.time_seconds !== undefined) {
      var t =
        typeof data.time_seconds === "number"
          ? data.time_seconds.toFixed(3)
          : String(data.time_seconds);
      parts.push("t=" + t + "s");
    }
    if (data.frame_index !== null && data.frame_index !== undefined) {
      parts.push("f=" + String(data.frame_index));
    }
    if (data.kind === "polyline" && data.points && data.points.length) {
      parts.push(String(data.points.length) + " vertices");
    } else {
      var p0 = data.points && data.points[0];
      if (p0) {
        parts.push(
          "(" +
            Number(p0.x_norm).toFixed(4) +
            ", " +
            Number(p0.y_norm).toFixed(4) +
            ")"
        );
      }
    }
    return parts.join(" · ");
  }

  /**
   * @param {HTMLElement} wrap
   * @param {Array<{ x_norm: number, y_norm: number }>} arr
   */
  function setQuadratPolygonContents(wrap, arr) {
    while (wrap.firstChild) {
      wrap.removeChild(wrap.firstChild);
    }
    if (!arr || arr.length === 0) {
      return;
    }
    var svgNS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 1 1");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("class", "annotation-polyline-svg calibration-quadrat-svg is-draft");
    if (arr.length < 4) {
      var pl = document.createElementNS(svgNS, "polyline");
      pl.setAttribute("fill", "none");
      var pair = [];
      for (var i = 0; i < arr.length; i++) {
        pair.push(arr[i].x_norm + " " + arr[i].y_norm);
      }
      pl.setAttribute("points", pair.join(" "));
      svg.appendChild(pl);
    } else {
      var pg = document.createElementNS(svgNS, "polygon");
      var pts = arr
        .map(function (p) {
          return p.x_norm + "," + p.y_norm;
        })
        .join(" ");
      pg.setAttribute("points", pts);
      pg.setAttribute("fill", "none");
      svg.appendChild(pg);
    }
    wrap.appendChild(svg);
    for (var j = 0; j < arr.length; j++) {
      var vtx = document.createElement("div");
      vtx.className =
        "annotation-polyline-vertex annotation-polyline-vertex-draft calibration-quadrat-vertex";
      vtx.style.left = arr[j].x_norm * 100 + "%";
      vtx.style.top = arr[j].y_norm * 100 + "%";
      wrap.appendChild(vtx);
    }
  }

  /**
   * @param {HTMLElement} draftEl
   * @param {HTMLVideoElement | HTMLImageElement} media
   * @param {Array<{ x: number, y: number }>} points
   */
  function renderQuadratDraftEl(draftEl, media, points) {
    if (!draftEl) {
      return;
    }
    var arr = points.map(function (p) {
      return { x_norm: p.x, y_norm: p.y };
    });
    positionPictureLayer(draftEl, media);
    if (arr.length === 0) {
      while (draftEl.firstChild) {
        draftEl.removeChild(draftEl.firstChild);
      }
      return;
    }
    setQuadratPolygonContents(draftEl, arr);
  }

  /**
   * Quadrat calibration: four clicks, then POST /media/{id}/calibration.
   * @param {HTMLElement} wrap — [data-annotation-video-root] or [data-annotation-image-root]
   * @param {HTMLElement} host — [data-media-video-viewer] or [data-media-image-viewer]
   * @param {HTMLElement} panel — [data-calibration-panel]
   * @param {{ overlay: HTMLElement, media: HTMLVideoElement | HTMLImageElement, mediaId: string, isVideo: boolean, video: HTMLVideoElement | null, fps: number, hasFps: boolean }} opts
   */
  function bindCalibrationUI(wrap, host, panel, opts) {
    var overlay = opts.overlay;
    var media = opts.media;
    var mediaId = opts.mediaId;
    var isVideo = opts.isVideo;
    var video = opts.video;
    var fps = opts.fps;
    var hasFps = opts.hasFps;

    var toggle = panel.querySelector("[data-calibration-toggle]");
    var after = panel.querySelector("[data-calibration-after-points]");
    var submitBtn = panel.querySelector("[data-calibration-submit]");
    var clearBtn = panel.querySelector("[data-calibration-clear-draft]");
    var errEl = panel.querySelector("[data-calibration-error]");

    var quadratDraft = [];
    var calibDraftEl = null;

    function showCalErr(msg) {
      if (!errEl) {
        return;
      }
      errEl.textContent = msg || "";
      errEl.hidden = !msg;
    }

    function clearCalibrationDraft() {
      quadratDraft = [];
      if (calibDraftEl && calibDraftEl.parentNode) {
        calibDraftEl.parentNode.removeChild(calibDraftEl);
      }
      calibDraftEl = null;
      if (after) {
        after.hidden = true;
      }
      showCalErr("");
    }

    function ensureCalibDraftEl() {
      if (calibDraftEl) {
        return calibDraftEl;
      }
      calibDraftEl = document.createElement("div");
      calibDraftEl.className = "annotation-polyline-wrap calibration-quadrat-draft";
      calibDraftEl.setAttribute("data-calibration-quadrat-draft", "1");
      overlay.appendChild(calibDraftEl);
      return calibDraftEl;
    }

    host.addEventListener("crabspy-reset-calibration", function () {
      host.classList.remove("is-calibrating-quadrat");
      if (toggle) {
        toggle.textContent = "Calibrate quadrat";
      }
      if (overlay) {
        overlay.setAttribute("aria-hidden", "true");
      }
      clearCalibrationDraft();
    });

    if (toggle) {
      toggle.addEventListener("click", function () {
        host.classList.toggle("is-calibrating-quadrat");
        var on = host.classList.contains("is-calibrating-quadrat");
        toggle.textContent = on ? "Cancel quadrat" : "Calibrate quadrat";
        if (!on) {
          clearCalibrationDraft();
          if (overlay) {
            overlay.setAttribute("aria-hidden", "true");
          }
        } else {
          wrap.classList.remove("is-annotating");
          var annToggle = host.querySelector("[data-annotation-toggle]");
          if (annToggle) {
            annToggle.textContent = "Annotate";
          }
          /* Keep the current frame; otherwise the first click can reach the video and start playback. */
          if (isVideo && video) {
            try {
              video.pause();
            } catch (e) {
              /* ignore */
            }
          }
          if (overlay) {
            overlay.setAttribute("aria-hidden", "false");
          }
        }
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        clearCalibrationDraft();
      });
    }

    overlay.addEventListener(
      "click",
      function calibCap(e) {
        if (!host.classList.contains("is-calibrating-quadrat")) {
          return;
        }
        e.preventDefault();
        e.stopPropagation();
        var pt = normalizedMediaClick(media, e.clientX, e.clientY);
        if (!pt) {
          return;
        }
        if (quadratDraft.length >= 4) {
          return;
        }
        quadratDraft.push({ x: pt.x, y: pt.y });
        ensureCalibDraftEl();
        renderQuadratDraftEl(calibDraftEl, media, quadratDraft);
        if (quadratDraft.length === 4 && after) {
          after.hidden = false;
        }
      },
      true
    );

    if (submitBtn) {
      submitBtn.addEventListener("click", function () {
        if (quadratDraft.length !== 4) {
          showCalErr("Place four corners first.");
          return;
        }
        var refMmEl = panel.querySelector("[data-calibration-ref-mm]");
        var edgeSel = panel.querySelector("[data-calibration-edge-idx]");
        var labEl = panel.querySelector("[data-calibration-label]");
        var mm = refMmEl && refMmEl.value ? parseFloat(refMmEl.value) : NaN;
        if (!Number.isFinite(mm) || mm <= 0) {
          showCalErr("Enter a positive reference length in mm.");
          return;
        }
        var edgeIdx = edgeSel ? parseInt(edgeSel.value, 10) : 0;
        var rd = getRefDimensions(media);
        if (!rd.ref_width_px || !rd.ref_height_px) {
          showCalErr("Wait for the image or video to load (dimensions unknown).");
          return;
        }
        var payload = {
          corners: quadratDraft.map(function (p) {
            return { x_norm: p.x, y_norm: p.y };
          }),
          reference_edge_index: edgeIdx,
          reference_length_mm: mm,
          ref_width_px: rd.ref_width_px,
          ref_height_px: rd.ref_height_px,
        };
        if (labEl && labEl.value && String(labEl.value).trim()) {
          payload.label = String(labEl.value).trim();
        }
        if (isVideo && video) {
          var t = video.currentTime;
          payload.time_seconds = t;
          if (hasFps) {
            payload.frame_index = Math.floor(t * fps);
          }
        }
        showCalErr("");
        fetch("/media/" + mediaId + "/calibration", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        })
          .then(function (r) {
            if (!r.ok) {
              return r.text().then(function (txt) {
                throw new Error(txt || "Could not save calibration.");
              });
            }
            return r.json();
          })
          .then(function () {
            window.location.reload();
          })
          .catch(function (err) {
            showCalErr(err.message || "Could not save calibration.");
          });
      });
    }

    document.addEventListener("keydown", function escCalib(e) {
      if (e.key !== "Escape") {
        return;
      }
      var tag = e.target && e.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
        return;
      }
      if (!host.classList.contains("is-calibrating-quadrat")) {
        return;
      }
      if (quadratDraft.length === 0) {
        return;
      }
      clearCalibrationDraft();
    });

    function syncCalib() {
      if (calibDraftEl && quadratDraft.length) {
        renderQuadratDraftEl(calibDraftEl, media, quadratDraft);
      }
    }

    if (media.tagName === "VIDEO") {
      media.addEventListener("loadeddata", syncCalib);
    }
    window.addEventListener("resize", syncCalib);
  }

  /**
   * @param {HTMLElement} host — [data-media-video-viewer] or [data-media-image-viewer]
   */
  function initCalibrationOnHost(host) {
    var panel = host.querySelector("[data-calibration-panel]");
    if (!panel) {
      return;
    }
    var wrap = host.querySelector("[data-annotation-video-root], [data-annotation-image-root]");
    if (!wrap) {
      return;
    }
    var overlay = wrap.querySelector("[data-annotation-overlay]");
    var video = wrap.querySelector("video");
    var img = wrap.querySelector("img");
    var media = video || img;
    if (!overlay || !media) {
      return;
    }
    var mediaId = wrap.getAttribute("data-media-id");
    if (!mediaId) {
      return;
    }
    var fpsRaw = host.getAttribute("data-video-fps");
    var fps =
      fpsRaw && String(fpsRaw).trim() !== ""
        ? parseFloat(fpsRaw)
        : NaN;
    var hasFps = Number.isFinite(fps) && fps > 0;
    bindCalibrationUI(wrap, host, panel, {
      overlay: overlay,
      media: media,
      mediaId: mediaId,
      isVideo: video !== null,
      video: video,
      fps: fps,
      hasFps: hasFps,
    });
  }

  /**
   * @param {HTMLElement} wrap — video or image annotation root
   * @param {HTMLElement} host — [data-media-video-viewer] or [data-media-image-viewer]
   * @param {{ media: HTMLVideoElement | HTMLImageElement, overlay: HTMLElement, list: HTMLElement | null, mediaId: string, isVideo: boolean, video: HTMLVideoElement | null, fps: number, hasFps: boolean }} opts
   */
  function bindAnnotationUI(wrap, host, opts) {
    var media = opts.media;
    var overlay = opts.overlay;
    var list = opts.list;
    var mediaId = opts.mediaId;
    var isVideo = opts.isVideo;
    var video = opts.video;
    var fps = opts.fps;
    var hasFps = opts.hasFps;

    var toggle = host.querySelector("[data-annotation-toggle]");
    var finishBtn = host.querySelector("[data-annotation-finish-polyline]");
    var cancelBtn = host.querySelector("[data-annotation-cancel-polyline]");
    var modeInputs = host.querySelectorAll("[data-annotation-mode-input]");

    var polylineDraft = [];
    var draftEl = null;

    function getMode() {
      if (!modeInputs || !modeInputs.length) {
        return "point";
      }
      for (var i = 0; i < modeInputs.length; i++) {
        if (modeInputs[i].checked) {
          return modeInputs[i].value;
        }
      }
      return "point";
    }

    function updatePolylineButtons() {
      var m = getMode();
      var drafting = polylineDraft.length > 0;
      if (finishBtn) {
        finishBtn.hidden = !(m === "polyline" && drafting && polylineDraft.length >= 2);
      }
      if (cancelBtn) {
        cancelBtn.hidden = !(m === "polyline" && drafting);
      }
    }

    function clearPolylineDraft() {
      polylineDraft = [];
      if (draftEl && draftEl.parentNode) {
        draftEl.parentNode.removeChild(draftEl);
      }
      draftEl = null;
      updatePolylineButtons();
      syncAll();
    }

    function ensureDraftEl() {
      if (draftEl) {
        return draftEl;
      }
      draftEl = document.createElement("div");
      draftEl.className = "annotation-polyline-wrap annotation-polyline-draft";
      draftEl.setAttribute("data-polyline-draft", "1");
      overlay.appendChild(draftEl);
      return draftEl;
    }

    function syncAll() {
      renderDots(overlay, media);
      var wraps = overlay.querySelectorAll(".annotation-polyline-wrap:not(.annotation-polyline-draft)");
      for (var i = 0; i < wraps.length; i++) {
        renderPolylineWrapFromData(wraps[i], media);
      }
      if (draftEl && polylineDraft.length) {
        renderPolylineDraftEl(draftEl, media, polylineDraft);
      }
    }

    if (toggle) {
      toggle.addEventListener("click", function () {
        wrap.classList.toggle("is-annotating");
        var on = wrap.classList.contains("is-annotating");
        toggle.textContent = on ? "Stop annotating" : "Annotate";
        if (on) {
          try {
            host.dispatchEvent(new CustomEvent("crabspy-reset-calibration"));
          } catch (e) {
            /* ignore */
          }
        }
        if (!on) {
          clearPolylineDraft();
        }
      });
    }

    if (modeInputs && modeInputs.length) {
      for (var mi = 0; mi < modeInputs.length; mi++) {
        modeInputs[mi].addEventListener("change", function () {
          clearPolylineDraft();
        });
      }
    }

    if (cancelBtn) {
      cancelBtn.addEventListener("click", function () {
        clearPolylineDraft();
      });
    }

    if (finishBtn) {
      finishBtn.addEventListener("click", function () {
        if (polylineDraft.length < 2) {
          return;
        }
        var payload = {
          kind: "polyline",
          points: polylineDraft.map(function (p) {
            return { x_norm: p.x, y_norm: p.y };
          }),
        };
        if (isVideo && video) {
          var time = video.currentTime;
          payload.time_seconds = time;
          if (hasFps) {
            payload.frame_index = Math.floor(time * fps);
          }
        }
        applyAnnotationPayloadExtras(host, media, payload);
        fetch("/media/" + mediaId + "/annotations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        })
          .then(function (r) {
            if (!r.ok) {
              throw new Error("save");
            }
            return r.json();
          })
          .then(function (data) {
            clearPolylineDraft();
            annotationAddPolylineWrap(overlay, media, data);
            if (list) {
              var li = document.createElement("li");
              li.setAttribute("data-annotation-id", data.id);
              li.appendChild(document.createTextNode(annotationFormatRow(data) + " "));
              var del = document.createElement("button");
              del.type = "button";
              del.className = "btn btn-danger annotation-delete";
              del.textContent = "Delete";
              li.appendChild(del);
              list.appendChild(li);
            }
          })
          .catch(function () {});
      });
    }

    document.addEventListener("keydown", function escPolyline(e) {
      if (e.key !== "Escape") {
        return;
      }
      var tag = e.target && e.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
        return;
      }
      if (!wrap.classList.contains("is-annotating")) {
        return;
      }
      if (getMode() !== "polyline" || polylineDraft.length === 0) {
        return;
      }
      clearPolylineDraft();
    });

    function syncMedia() {
      syncAll();
    }

    if (media.tagName === "VIDEO") {
      if (media.readyState >= HAVE_METADATA) {
        syncMedia();
      } else {
        media.addEventListener("loadedmetadata", syncMedia, { once: true });
      }
      media.addEventListener("loadeddata", syncMedia);
    } else if (media.tagName === "IMG") {
      if (media.complete && media.naturalWidth) {
        syncMedia();
      } else {
        media.addEventListener("load", syncMedia, { once: true });
      }
    }
    window.addEventListener("resize", syncMedia);

    overlay.addEventListener("click", function (e) {
      if (host.classList.contains("is-calibrating-quadrat")) {
        return;
      }
      if (!wrap.classList.contains("is-annotating")) {
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      var pt = normalizedMediaClick(media, e.clientX, e.clientY);
      if (!pt) {
        return;
      }
      var mode = getMode();
      if (mode === "point") {
        var payload = {
          kind: "point",
          points: [{ x_norm: pt.x, y_norm: pt.y }],
        };
        if (isVideo && video) {
          var t = video.currentTime;
          payload.time_seconds = t;
          if (hasFps) {
            payload.frame_index = Math.floor(t * fps);
          }
        }
        applyAnnotationPayloadExtras(host, media, payload);
        fetch("/media/" + mediaId + "/annotations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        })
          .then(function (r) {
            if (!r.ok) {
              throw new Error("save");
            }
            return r.json();
          })
          .then(function (data) {
            annotationAddDot(overlay, media, data);
            if (list) {
              var li = document.createElement("li");
              li.setAttribute("data-annotation-id", data.id);
              li.appendChild(document.createTextNode(annotationFormatRow(data) + " "));
              var del = document.createElement("button");
              del.type = "button";
              del.className = "btn btn-danger annotation-delete";
              del.textContent = "Delete";
              li.appendChild(del);
              list.appendChild(li);
            }
          })
          .catch(function () {});
        return;
      }
      polylineDraft.push({ x: pt.x, y: pt.y });
      ensureDraftEl();
      renderPolylineDraftEl(draftEl, media, polylineDraft);
      updatePolylineButtons();
    });

    if (list) {
      list.addEventListener("click", function (e) {
        var t = e.target;
        if (!t || !t.classList || !t.classList.contains("annotation-delete")) {
          return;
        }
        var li = t.closest("li");
        var aid = li && li.getAttribute("data-annotation-id");
        if (!aid) {
          return;
        }
        fetch("/media/" + mediaId + "/annotations/" + aid + "/delete", {
          method: "POST",
        })
          .then(function (r) {
            if (!r.ok) {
              throw new Error("del");
            }
            removeAnnotationVisual(overlay, aid);
            if (li && li.parentNode) {
              li.parentNode.removeChild(li);
            }
          })
          .catch(function () {});
      });
    }
  }

  /**
   * @param {HTMLElement} wrap — [data-annotation-video-root]
   */
  function initAnnotationVideo(wrap) {
    var video = wrap.querySelector("video");
    var overlay = wrap.querySelector("[data-annotation-overlay]");
    if (!video || !overlay) {
      return;
    }
    var host = wrap.closest("[data-media-video-viewer]");
    if (!host) {
      return;
    }
    var list = host.querySelector(".annotation-list");
    var mediaId = wrap.getAttribute("data-media-id");
    if (!mediaId) {
      return;
    }
    var fpsRaw = host.getAttribute("data-video-fps");
    var fps =
      fpsRaw && String(fpsRaw).trim() !== ""
        ? parseFloat(fpsRaw)
        : NaN;
    var hasFps = Number.isFinite(fps) && fps > 0;

    bindAnnotationUI(wrap, host, {
      media: video,
      overlay: overlay,
      list: list,
      mediaId: mediaId,
      isVideo: true,
      video: video,
      fps: fps,
      hasFps: hasFps,
    });
  }

  /**
   * @param {HTMLElement} wrap — [data-annotation-image-root]
   */
  function initAnnotationImage(wrap) {
    var img = wrap.querySelector("img");
    var overlay = wrap.querySelector("[data-annotation-overlay]");
    if (!img || !overlay) {
      return;
    }
    var host = wrap.closest("[data-media-image-viewer]");
    if (!host) {
      return;
    }
    var list = host.querySelector(".annotation-list");
    var mediaId = wrap.getAttribute("data-media-id");
    if (!mediaId) {
      return;
    }

    bindAnnotationUI(wrap, host, {
      media: img,
      overlay: overlay,
      list: list,
      mediaId: mediaId,
      isVideo: false,
      video: null,
      fps: NaN,
      hasFps: false,
    });
  }

  function init() {
    var nodes = document.querySelectorAll("[data-media-video-viewer]");
    for (var i = 0; i < nodes.length; i++) {
      initMediaVideoViewer(nodes[i]);
    }
    var annRoots = document.querySelectorAll("[data-annotation-video-root]");
    for (var j = 0; j < annRoots.length; j++) {
      initAnnotationVideo(annRoots[j]);
    }
    var imgRoots = document.querySelectorAll("[data-annotation-image-root]");
    for (var k = 0; k < imgRoots.length; k++) {
      initAnnotationImage(imgRoots[k]);
    }
    var calHosts = document.querySelectorAll(
      "[data-media-video-viewer], [data-media-image-viewer]"
    );
    for (var c = 0; c < calHosts.length; c++) {
      initCalibrationOnHost(calHosts[c]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

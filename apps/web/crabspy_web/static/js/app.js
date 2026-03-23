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
   * Returns picture rectangle inside the rendered video element (letterbox excluded).
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
   * Map viewport click to normalized coords (0–1) within the visible video picture (letterbox excluded).
   * @returns {{ x: number, y: number } | null}
   */
  function normalizedVideoClick(video, clientX, clientY) {
    var rect = video.getBoundingClientRect();
    var relX = clientX - rect.left;
    var relY = clientY - rect.top;
    var pic = getVideoPictureRect(video);
    if (relX < pic.x0 || relX > pic.x0 + pic.width || relY < pic.y0 || relY > pic.y0 + pic.height) {
      return null;
    }
    return { x: (relX - pic.x0) / pic.width, y: (relY - pic.y0) / pic.height };
  }

  function placeDot(dot, video) {
    var xNorm = parseFloat(dot.getAttribute("data-x-norm") || "");
    var yNorm = parseFloat(dot.getAttribute("data-y-norm") || "");
    if (!Number.isFinite(xNorm) || !Number.isFinite(yNorm)) {
      return;
    }
    var pic = getVideoPictureRect(video);
    dot.style.left = pic.x0 + xNorm * pic.width + "px";
    dot.style.top = pic.y0 + yNorm * pic.height + "px";
  }

  function renderDots(overlay, video) {
    var dots = overlay.querySelectorAll(".video-spike-dot");
    for (var i = 0; i < dots.length; i++) {
      placeDot(dots[i], video);
    }
  }

  function videoSpikeAddDot(overlay, video, data) {
    var dot = document.createElement("div");
    dot.className = "video-spike-dot";
    dot.setAttribute("data-point-id", data.id);
    dot.setAttribute("data-x-norm", String(data.x_norm));
    dot.setAttribute("data-y-norm", String(data.y_norm));
    overlay.appendChild(dot);
    placeDot(dot, video);
  }

  function videoSpikeFormatRow(data) {
    var t = typeof data.time_seconds === "number" ? data.time_seconds.toFixed(3) : String(data.time_seconds);
    var parts = ["t=" + t + "s"];
    if (data.frame_index !== null && data.frame_index !== undefined) {
      parts.push("f=" + String(data.frame_index));
    }
    parts.push(
      "(" +
        Number(data.x_norm).toFixed(4) +
        ", " +
        Number(data.y_norm).toFixed(4) +
        ")"
    );
    return parts.join(" · ");
  }

  /**
   * @param {HTMLElement} wrap — [data-video-spike-root]
   */
  function initVideoSpike(wrap) {
    var video = wrap.querySelector("video");
    var overlay = wrap.querySelector("[data-video-spike-overlay]");
    if (!video || !overlay) {
      return;
    }
    var host = wrap.closest("[data-media-video-viewer]");
    if (!host) {
      return;
    }
    var toggle = host.querySelector("[data-video-spike-toggle]");
    var list = host.querySelector(".video-spike-list");
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

    if (toggle) {
      toggle.addEventListener("click", function () {
        wrap.classList.toggle("is-annotating");
        var on = wrap.classList.contains("is-annotating");
        toggle.textContent = on ? "Stop adding points" : "Add annotation points";
      });
    }

    function syncDots() {
      renderDots(overlay, video);
    }

    if (video.readyState >= HAVE_METADATA) {
      syncDots();
    } else {
      video.addEventListener("loadedmetadata", syncDots, { once: true });
    }
    video.addEventListener("loadeddata", syncDots);
    window.addEventListener("resize", syncDots);

    overlay.addEventListener("click", function (e) {
      if (!wrap.classList.contains("is-annotating")) {
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      var pt = normalizedVideoClick(video, e.clientX, e.clientY);
      if (!pt) {
        return;
      }
      var time = video.currentTime;
      var payload = {
        x_norm: pt.x,
        y_norm: pt.y,
        time_seconds: time,
      };
      if (hasFps) {
        payload.frame_index = Math.floor(time * fps);
      }
      fetch("/media/" + mediaId + "/video-spike", {
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
          videoSpikeAddDot(overlay, video, data);
          if (!list) {
            return;
          }
          var li = document.createElement("li");
          li.setAttribute("data-point-id", data.id);
          li.appendChild(document.createTextNode(videoSpikeFormatRow(data) + " "));
          var del = document.createElement("button");
          del.type = "button";
          del.className = "btn btn-danger video-spike-delete";
          del.textContent = "Delete";
          li.appendChild(del);
          list.appendChild(li);
        })
        .catch(function () {});
    });

    if (list) {
      list.addEventListener("click", function (e) {
        var t = e.target;
        if (!t || !t.classList || !t.classList.contains("video-spike-delete")) {
          return;
        }
        var li = t.closest("li");
        var pid = li && li.getAttribute("data-point-id");
        if (!pid) {
          return;
        }
        fetch("/media/" + mediaId + "/video-spike/" + pid + "/delete", {
          method: "POST",
        })
          .then(function (r) {
            if (!r.ok) {
              throw new Error("del");
            }
            var c = overlay.querySelector('[data-point-id="' + pid + '"]');
            if (c && c.parentNode) {
              c.parentNode.removeChild(c);
            }
            if (li && li.parentNode) {
              li.parentNode.removeChild(li);
            }
          })
          .catch(function () {});
      });
    }
  }

  function init() {
    var nodes = document.querySelectorAll("[data-media-video-viewer]");
    for (var i = 0; i < nodes.length; i++) {
      initMediaVideoViewer(nodes[i]);
    }
    var spikes = document.querySelectorAll("[data-video-spike-root]");
    for (var j = 0; j < spikes.length; j++) {
      initVideoSpike(spikes[j]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

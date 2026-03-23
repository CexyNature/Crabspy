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

  function init() {
    var nodes = document.querySelectorAll("[data-media-video-viewer]");
    for (var i = 0; i < nodes.length; i++) {
      initMediaVideoViewer(nodes[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

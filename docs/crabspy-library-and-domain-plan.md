# Crabspy library rebuild and domain feature plan

This document complements [`crabspy-rebuild-plan.md`](crabspy-rebuild-plan.md). It treats **rebuilding and organizing `crabspy/`** as a **first-class prerequisite** for Phase 4+ processing, and maps your **biology / CV workflows** into a maintainable architecture and phased delivery.

**Script-by-script inventory:** [`legacy-inventory.md`](legacy-inventory.md) (paths, proposed modules, DB touchpoints, and priority).

## 1. Why rebuild `crabspy/` first

Today, `crabspy/` is largely a **collection of scripts** (argparse entrypoints, ad hoc `results/` paths, pickle/CSV sidecars, tight coupling to `methods.py` / `constant.py`). That was appropriate for research but blocks:

- **Testing** (few pure functions, hard to mock I/O).
- **Reuse** from FastAPI, CLI, and batch jobs with the same behavior.
- **Clear contracts** (inputs/outputs, units, coordinate spaces).
- **Evolution** as you add tracking, calibration, and database-backed identities.

**Goal:** a **versioned, installable library** (under `crabspy/` or `packages/crabspy`) with:

| Layer | Responsibility |
| ----- | ---------------- |
| **Core types** | Frame references (video id, frame index, time), image/video handles, normalized vs pixel geometry, units (px, mm). |
| **Geometry** | Polylines, rectangles, masks; transforms between normalized display coords and source pixels. |
| **Calibration** | Known-length reference (e.g. quadrat) → pixel scale (mm/px) or homography if needed later. |
| **Appearance / background** | Background model from frame series; foreground masks; optional morphological cleanup. |
| **Detection & tracking** | Motion-based or feature-based association; track IDs over time. |
| **Measurement** | Carapace polylines + scale → mm; uncertainty policy documented. |
| **Color / profiles** | Histograms or compact signatures per blob/track; storage format. |
| **Persistence adapters** | Implementations that read/write **your SQL schema** (via `crabspy_web` models or a shared `crabspy.persistence` port), not pickle in arbitrary folders. |

Legacy scripts (`measure.py`, `get_color.py`, `track*.py`, `map_burrows*.py`, etc.) become **thin CLIs** calling this library—or are retired once the web UI covers the workflow.

---

## 2. Domain features → technical ingredients

Below is a **feature matrix**: what each capability needs in the **library**, **database**, and **web UI**.

### 2.1 Carapace annotation (polyline on images / video frames)

- **Library:** polyline in pixel or normalized space; optional simplification; length in px and, if calibrated, mm.
- **DB:** Already have `annotation` + `annotation_point` + time/frame for video; may need **semantic type** (e.g. `annotation.purpose = carapace | calibration_frame | other`) or separate table if calibration quadrats have different rules.
- **UI:** Done at basic level (Phase 3b); may need **mode** (“carapace” vs “calibration rectangle”) and snapping/validation.

### 2.2 Calibration: known-length rectangle in one video → real size (mm)

- **Library:**  
  - User provides **four corners** (rectangle or general quadrilateral in image space).  
  - Compute **scale** along relevant axes (isotropic first: single mm/px; later anisotropic or homography if perspective matters).  
  - Store **reference length(s)** (mm) and **which edge(s)** define scale.
- **DB:** `calibration` or fields on `media` / dedicated table: `reference_length_mm`, corner points (or `annotation_id` FK to the calibration annotation), validity, created_at.
- **UI:** Dedicated tool to draw/mark rectangle on **one** representative frame; persist link to **frame index** and **media_id**.

### 2.3 Link every crab measurement to frame, video metadata, unique ID, species

- **DB:** Introduce **`individual`** or **`specimen`** (unique id, species, optional labels) and **`observation`** / **`measurement_event`** linking:  
  `media_id`, `frame_index` or `time_seconds`, `individual_id`, `annotation_id` (carapace polyline), derived mm, etc.  
  Species can be enum/table + free text where needed.
- **Library:** Pure functions: given annotation + calibration record → mm; no duplicate business logic in UI.
- **UI:** Forms and list views; filters by species, video, date (from media metadata).

### 2.4 “Optimal frame” for annotation, saved in database

- **DB:** `key_frame` or flag on annotation: `is_canonical` / `preferred_frame_index` per (individual, video) or per measurement session—**schema decision**: one canonical frame per crab per video vs many candidates.
- **UI:** Scrub timeline (you have seek); **“Set as key frame”** persists index (+ optional thumbnail path in cache).

### 2.5 Background model from frame series; motion masking for crabs

- **Library:**  
  - `BackgroundModel` API (e.g. MOG2, median frame stack, or simpler running average).  
  - Input: list of frame indices or time range; output: model artifact path + metadata.  
  - `foreground_mask(frame, model)` → binary mask; optional crab-sized connected components.
- **DB:** Job record + path to serialized model or parameters; link to `media_id`.
- **UI:** Phase 4-style job: select frame range → **Train background** → show preview overlay.

### 2.6 Tracking (motion or features) + relate tracks to annotations

- **Library:**  
  - Pluggable **tracker** (optical flow + bbox, KLT, CSRT, or simpler centroid linking—match legacy `track.py` / `fast_track.py` behavior where useful).  
  - Output: **track_id**, per-frame bbox or polygon, optional confidence.
- **DB:** `track`, `track_point` (frame_index, x, y, w, h, …); optional FK from **carapace annotation** to **track_id** for a given frame range.
- **UI:** Overlay tracks on video; jump to frame; export CSV (already a pattern in legacy `results/`).

### 2.7 Color profile for annotated crab blobs

- **Library:**  
  - Given mask or bbox + frame → histogram (Lab/HSV) or fixed-length feature vector; optional temporal aggregation along track.  
  - Version the feature schema for reproducibility.
- **DB:** `color_profile` table: `individual_id` or `annotation_id`, `track_id`, serialized vector or JSON + summary stats.
- **UI:** Trigger “Extract color profile” post-annotation or post-track; display simple viz (optional).

---

## 3. Phased implementation (recommended order)

Dependencies matter: **calibration before mm**, **frame identity before track↔annotation linking**, **background before motion masking** (for that pipeline).

| Phase | Focus | Delivers |
| ----- | ----- | -------- |
| **A. Library skeleton** | Package layout, `pyproject.toml`, core types, geometry, logging, pytest fixtures (tiny synthetic video/image). | Importable `crabspy` with no GUI; CI runs unit tests. |
| **B. Calibration + measurement** | Port logic from `measure.py` / quadrat handling in `get_color.py`-style metadata into **pure functions** + tests. | mm/px from reference rectangle + carapace polyline length in mm. |
| **C. Persistence port** | SQLAlchemy models or repository interfaces for individuals, species, calibration, key frames. | Web and CLI share one source of truth. |
| **D. Background job (vertical slice)** | One OpenCV path: train BG on frame range, save artifact, expose Phase 4 job UI. | End-to-end “processing” proof; aligns with rebuild Phase 4. |
| **E. Tracking port** | Refactor `track.py` / `fast_track.py` into library; stable CSV/DB output schema. | Tracks linkable to media + frame. |
| **F. Color profiles** | Port `get_color.py` ideas to mask-based histogram API; DB + optional UI. | Reproducible signatures per blob/track. |
| **G. Web UX consolidation** | Modes for carapace vs calibration, individual/species forms, key-frame button, track overlay. | Researchers rarely need raw scripts. |

**Annotation import from CSV** (deferred in Phase 3b) can land after **C** if you need migration from old `results/*.csv`.

---

## 4. Schema extensions (sketch)

Exact tables belong in Alembic migrations; conceptually you will likely need:

- **`taxon` / `species`** (or enum + text)
- **`individual`** (uuid, species_id, external_id, notes)
- **`calibration`** (media_id, frame_index, ref_length_mm, geometry FK or inline corners)
- **`media_key_frame`** (media_id, frame_index, purpose, optional individual_id)
- **`processing_job`** (status, kind, media_id, error, artifact paths) — Phase 4
- **`track` / `track_frame`**
- **`color_profile`**

Existing **`media`** + **`annotation`** remain the anchor; new tables **reference** them.

---

## 5. Risks and mitigations

| Risk | Mitigation |
| ---- | ---------- |
| **Scope explosion** | Keep Phase 4 to **one** pipeline; add others behind feature flags or phases E–G. |
| **Legacy CSV/pickle** | Write one-off **migration scripts** into SQL; don’t let dual-write linger. |
| **Perspective / lens distortion** | Start with **isotropic scale** from a reference rectangle; document error; add homography later if needed. |
| **Performance on long videos** | Chunked decode, optional frame cache dir, background jobs only. |
| **GPL and dependencies** | Keep third-party stack documented; separate optional heavy deps (e.g. deep models) if added later. |

---

## 6. Relation to `crabspy-rebuild-plan.md`

- **Phase 4** (single processing slice) should **use** the new library’s **Background** or **Calibration** slice once **A + part of B/D** exist—not ad hoc script copy-paste.
- **Phase 5** generalizes **jobs + storage**; tracking and color fit naturally as additional job types.

---

## 7. Immediate next steps (when you start execution)

1. **Inventory** each legacy script: inputs, outputs, side effects → map to library module in §1.  
2. **Freeze** one **reference workflow** (e.g. “quadrat + carapace length in mm”) as the **acceptance test** for B+C.  
3. **Create** minimal package structure + CI test running **import crabspy** + one geometry test.  
4. **Align** with `apps/web` models: either add tables in `crabspy_web` migrations or a shared package—**one** persistence story.

This document should be updated as phases complete (check off rows, adjust dependencies).

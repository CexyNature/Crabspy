# Legacy `crabspy/` inventory

This document inventories Python modules under [`crabspy/`](../crabspy/) (and nested `toolkit/`) as of the rebuild. It maps each script to a **proposed library area**, **database touchpoints** once the SQL app owns truth, and a **priority** for porting.

**Priority legend**

| Tag | Meaning |
| --- | ------- |
| **P0** | Foundation: split `methods`/`constant`, video I/O, geometry, calibration path—blocks most other ports. |
| **P1** | Core domain: measurement, quadrat scale, tracks as first-class data. |
| **P2** | Derivatives: color profiles, stats, visualization CLIs, deduplication. |
| **P3** | Specialized or parallel pipelines: scoop-feed behavior, SVM/HOG handedness, 3D tooling—port after shared abstractions exist. |

**Shared legacy dependencies**

- **`methods.py`** — Large grab-bag: `read_video`, logging, mouse callbacks, quadrat capture, file creation times, etc. Becomes multiple focused modules (`io.video`, `geometry.quadrat`, …).
- **`constant.py`** — User-tuned experiment constants; becomes **config** (files/env) or **defaults** in library + DB for per-project overrides.

---

## Inventory table

| Path | Role (short) | Typical legacy I/O | Proposed library module(s) | DB / app touchpoints | Priority |
| ---- | ------------- | ------------------- | --------------------------- | --------------------- | -------- |
| `constant.py` | Global resize, quadrat dims, colors, morph kernels, flags | Imported everywhere | `crabspy.config` / typed settings | Optional `project_settings` or env | **P0** |
| `methods.py` | Video read, logging, UI mouse drawing, quadrat vertices, helpers | `results/Log`, `video/` paths | Split: `io.video`, `geometry`, `ui.interaction` (optional), `logging` | Media paths align with `crabspy_web` storage | **P0** |
| `measure.py` | Frame navigation + measure objects; save measures CSV | `results/measures/*.csv` | `measurement.carapace`, `export.csv` | `annotation`, `media`, future `calibration`, `individual` | **P1** |
| `get_color.py` | Histogram / color from track + quadrat metadata in CSV | `results/*.csv`, matplotlib | `appearance.color_profile`, `calibration` (read quadrat) | `color_profile`, `track`, `annotation` | **P1** |
| `track.py` | Track **one** individual in video (bbox/polygon over time) | `results/*tracking*.csv` | `tracking.single`, `io.video` | `track`, `track_frame`, link to `media` + `individual` | **P1** |
| `fast_track.py` | Track with frame skipping; single/multiple tracker modes | `results/*.csv` | `tracking.sparse`, same as above | Same | **P1** |
| `track_manual.py` | Manual tracking with mouse | `results/*.csv` | `tracking.manual` | Same | **P1** |
| `track_draw.py` | Replay one track on video | reads CSV | `tracking.visualize` | Read-only from DB export | **P2** |
| `tracks_viz.py` | Replay **unified** tracks + metafile | CSV + video | `tracking.visualize` | Same | **P2** |
| `tracks_unify.py` | Merge multiple track CSVs for one video | `results/*.csv` | `tracking.merge` or ETL into SQL | Import job + `track` rows | **P2** |
| `duplicates.py` | Find/repair duplicate frame rows in track CSV | `results/*.csv` | `tracking.qa` | Integrity constraints + repair job | **P2** |
| `get_stats.py` | Summary stats for one track file | `results/*.csv` | `tracking.stats` | Aggregates from SQL / export | **P2** |
| `get_binary_video.py` | Binary/motion-only video from movement blobs | video out, `results/` | `background.motion_video` or `segmentation.motion` | `processing_job`, artifact paths | **P1** |
| `manual_tracking.py` | (Thin) manual tracking entry—opens bin data | paths under `results/` | Absorb into `tracking.manual` | Same as manual track | **P2** |
| `map_burrows.py` | Draw circles on video for burrow positions → CSV | `results/*.csv` | `ecology.burrows` *or* `annotation` kind | New annotation type or `burrow` table | **P3** (if still needed) |
| `map_burrows_from_image.py` | Burrow mapping from still image | CSV | Same | Same | **P3** |
| `get_individuals_hog.py` | HOG features from crab snapshot folders for ML | `results/snapshots/...` | `features.hog` | Paths → `media` / export | **P3** |
| `svm_hog.py` | Train SVM on HOG (handedness classifier) | `results/snapshots/SVM_LR`, `.sav` | `ml.handedness` (optional package extra) | Model registry / artifact storage | **P3** |
| `scoop_feed.py` | Feeding activity / skimage graph pipeline on video | plots, intermediate | `behavior.scoop_feed` (heavy deps) | Optional job type | **P3** |
| `scoop_feed_v1.py` | Variant of scoop feed | same | Same | Same | **P3** |
| `scoop_feed_predict.py` | Predict scoop-feed class | model files | Same | Same | **P3** |
| `extract_scoop_feed.py` | Manual classify claw_up / claw_down images | image folders | `behavior.scoop_feed.data` | Training sets as artifacts | **P3** |
| `train_scoop_feed.py` | Train SVM on scoop-feed HOG | `train_data/`, `.sav` | Same as `svm_hog` path | Same | **P3** |
| `snaps_dict.py` | Walk snapshots, build JSON index of crab images | `results/snapshots/`, JSON | ETL → DB or `media` children | Snapshot table or file registry | **P2** |
| `database.py` | CLI prune/list pickled crab name DB | pickle under `results/<video_name>` | **Replaced** by SQL; migration tool | `individual` / legacy import | **P2** (migration) |
| `test.py` | Experiments / commented `CrabNames` pickle API | pickle | Tests only | N/A | **P2** (fixtures) |
| `toolkit/list_videos.py` | Recurse folder, ffprobe metadata → `videos.csv` | `videos.csv` | `io.inventory` | Bulk `media` import | **P2** |
| `toolkit/3Dmodels/frames_extraction.py` | Extract frames for 3D pipeline | image sequence | `io.frames` | Optional | **P3** |
| `toolkit/3Dmodels/frames_extraction_batch.py` | Batch frame extraction | dirs | Same | Same | **P3** |
| `toolkit/3Dmodels/run_vsfm_batch.py` | VSfM batch (structure from motion) | external tools | `toolkit.sfm` (optional) | Rare | **P3** |

---

## Grouping by rebuild theme

### A. Foundation (P0)

| Item | Action |
| ---- | ------ |
| `constant.py` | Replace with config schema; no mutable globals in library code. |
| `methods.py` | Decompose into testable functions; remove hard-coded `results/Log` where possible. |

### B. Calibration + measurement (P1)

| Item | Action |
| ---- | ------ |
| `measure.py` | Port polylines + scale → `crabspy.measurement`; persist via web models + calibration table. |
| `get_color.py` | Depends on quadrat vertices in track CSV—re-express as **DB-linked calibration** + **mask from track**. |

### C. Background + motion (P1)

| Item | Action |
| ---- | ------ |
| `get_binary_video.py` | First-class **job**: params (frame range, threshold), output video path, metadata in `processing_job`. |

### D. Tracking (P1–P2)

| Item | Action |
| ---- | ------ |
| `track.py`, `fast_track.py`, `track_manual.py` | Unified **track writer** API (CSV interim → SQL). |
| `tracks_unify.py`, `duplicates.py`, `get_stats.py` | Become **QA/reporting** on normalized track schema. |
| `track_draw.py`, `tracks_viz.py` | Visualization uses **exported** data or reads DB through app. |

### E. Appearance / ML (P2–P3)

| Item | Action |
| ---- | ------ |
| `get_color.py` (core) | `appearance.color_profile` table + numpy histograms. |
| `get_individuals_hog.py`, `svm_hog.py` | Optional **`crabspy[ml]`** extra; model artifacts under `data/cache` or similar. |
| Scoop-feed suite | Isolate in `crabspy.behavior.scoop_feed` with heavy deps optional. |

### F. Ecology / niche (P3)

| Item | Action |
| ---- | ------ |
| `map_burrows*.py` | Either new **annotation kind** + export or small domain module if actively used. |

### G. Tooling / ops (P2–P3)

| Item | Action |
| ---- | ------ |
| `toolkit/list_videos.py` | Bulk register media + ffprobe metadata (aligns with Phase 2 media rows). |
| `database.py` | One-off **pickle → SQL** migration script; then retire. |

---

## Suggested port order (execution checklist)

1. **Freeze** a minimal **track CSV schema** (columns you must keep) and **measure CSV** for migration scripts.
2. **Extract** `read_video` + metadata from `methods.py` → `crabspy.io.video` + tests.
3. **Port** `measure.py` math paths → library + unit tests (synthetic images).
4. **Port** `get_binary_video.py` pipeline → library + Phase 4 job wrapper.
5. **Port** `track.py` / `fast_track.py` core loops → `crabspy.tracking` with injectable OpenCV trackers.
6. **Port** `get_color.py` analysis → `crabspy.appearance` once tracks exist in SQL or as numpy masks.
7. **Defer** scoop-feed, SVM, 3D toolkit until A–F stabilize.

---

## Maintenance

Update this file when:

- A legacy script is **deprecated** (mark “retired” and date).
- A module is **fully ported** (add “→ `crabspy.modname`” and link to tests).

See also: [`crabspy-library-and-domain-plan.md`](crabspy-library-and-domain-plan.md).

# Experiment code

Every script that runs a model or computes a number in the paper. The scripts are sorted by paper section (folder prefix = section number, `appK` = Appendix K). The plotting scripts for the figures are in the section folders one level up (`../sec*/`, `../appendix/figures/`).

The code is the original research code, copied from the working packs (`events_pack`, `handoff_steering`, `xfer_pack`, `xfer_sweep_*`, `paper_plan`). Comments and docstrings are in Chinese. Only the paths between scripts were changed. Every script that runs a model `exec`s the shared prefix of `00_lib/group.py`, which now loads by relative path. The shell runners `cd` into their own folder.

## Layout

| Folder | Paper | Runs a model? | Scripts |
|---|---|---|---|
| `00_lib/` | §2, App. B | yes | `group.py`: shared core (arithmetic tasks, model loading, hooks, per-layer shape change, identity probe, patching). It is also run directly per pair. `new_concepts.py`: retrieval tasks (city, continent, capital→continent, color, context color, emotion) |
| `02_setup/` | §2, App. B–D | yes | `fill_queue.sh` runs, per model, `dims.py` (stores answer-position activations and shape-change variants), `group.py`, `steer_true.py` (centroid-difference steering), `lens.py` (logit lens) and `multipos.py` (patching at all answer positions / last position / relay). Peak robustness (App. C): `shape_noise.py` (bootstrap noise floor), `shape_cv.py` (cross-validated shape change) |
| `03_events/` | §3, App. E | no | `events_v2.py`: event classification used in the paper (writes `signals_v2.csv`, `events_v2.csv`, `handoff_types_v2.csv`). `stats_v2.py`, `downstream_v2.py`: §3 statistics and threshold sweep. v1 classification, used for the smaller models: `event_types.py`, `event_sensitivity.py`, `event_plot.py`. `explore.py`: features of each peak type |
| `04_handoff/` | §4, App. F | yes | `steer_handoff.py` (+ `run_handoff.sh`): input-token vs. answer-position steering per layer. `handoff_plot.py`: summary. `regroup.py`: handoff layer vs. peaks by event type (writes `regroup.csv`, used by `top3.py`, `verify_patch.py`, `../05_onset/sweep_cross.py`). `peak_perm.py`: circular-shift null for handoff and onset. `top3.py`, `verify_patch.py`: internal checks, not reported |
| `05_onset/` | §5, App. G–H | yes | `steer_sweep.py` (+ `sweep_run.sh` multi-GPU, `sweep_queue.sh` single GPU): strength sweep α = 1, 2, 4, 8. `sweep_report.py`: onsets per strength (`report_onset.csv`). `existing_onset.py`, `sweep_cross.py`: onsets from the other signals. App. H (logit-lens accessibility): `lap_compare.py`, `appendix_tables.py` |
| `06_transfer/` | §6, App. I | yes | `steer_xfer.py` (+ `xfer_queue.sh`, `xfer_pairs.sh`, `xfer_worker.sh`): build at layer X, apply at layer Y. `xfer_analyze.py`, `xfer_report.py`, `xfer_extra.py`: transfer strength, largest rise, permutation test. `xfer_grid.py`: overview heatmap. `dir_cos.py`: cross-layer cosine of centroid directions |
| `07_theory/` | §6.3, §7, App. J | no | `joint_check.py`: joint fit of inputs and answer (the version in the paper; `--synthetic` for recovery checks). `conv_check.py` → `conv_analyze.py`: blurred-input model. `weekday_noise.py`: noise floor of the weekday prediction. `fisher_xfer_check.py`: Fisher ratio vs. required strength, and transfer vs. cumulative shape change (§6.3). Outputs are in `../data/theory/` |
| `appK_lowacc/` | App. K | no | `lowacc_v2.py`: main numbers without the four low-accuracy pairs |

## Order

1. `02_setup/fill_queue.sh` for each model. The model family is chosen with `MK` (`llama`, `qwen`, `gemma`, `mistral`).
2. `04_handoff/run_handoff.sh`, `05_onset/sweep_run.sh`, `06_transfer/xfer_queue.sh` (independent; GPU).
3. `03_events/events_v2.py`, then the analysis scripts of each section. `../main_table/table_main.py` builds Table 1 from the resulting CSVs.

## Paths and missing pieces

- All scripts read and write under `paths.DATA` (the package's `data/`, or `$SHAPE_DATA`). The folder names under `data/` are the original `results/<name>/` folders; `group.py` writes to `data/group/<model>/<task>/ans/` and `dims.py` saves activations to `data/acts/`.
- `run_model.py` runs every GPU experiment for a model (replaces the per-step queues `fill_queue.sh`, `sweep_run.sh`, `xfer_queue.sh`, `run_handoff.sh`, which are kept for reference).
- **Rewritten in v2 (originals lost):** `04_handoff/handoff_patch.py` (input-token vs. answer-position patching; retrieval-task steering) and `02_setup/lens_full.py` (full-vocabulary logit lens); validated on the original Qwen2.5-7B with `04_handoff/validate_rewrite.py` (see `../CHANGES.md`).
- **Still not available:** the input-side decomposition of App. F (`handoff_decomp/`), the activations of the smaller models used in §7, `depth_vs_shape.csv`; `04_handoff/regroup.py` and `verify_patch.py` read files from the lost `handoff_tentative_paper/` folder.
- `05_onset/sweep_queue.sh` calls a model-download helper that is not included; use `run_model.py` or `sweep_run.sh`.

# Where the Answer Takes Shape — code, data and figures (v2: six models)

> **GitHub version.** This repository omits the activations in `data/acts/` (1.7 GB, see `data/acts/README.md`). Everything else, including all other inputs and outputs, is the same as the full release.


Code and data for the paper, now covering **six instruction-tuned models (59 model–task pairs)**:
Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9B-it (the original four), and
**Qwen2.5-14B-Instruct** and **Phi-4 (14B)** (added in v2). What changed since the first release is listed in `CHANGES.md`.

Requires Python 3, numpy, pandas, scipy, scikit-learn, matplotlib, torch, transformers, huggingface_hub
(tested with Python 3.12, torch 2.8.0, transformers 5.18.0).

## Quick start

```bash
bash reproduce.sh          # CPU only, ~3 min: every statistic, table and figure of the paper from data/
```

`reproduce.sh` recomputes the event classification, Table 1, the number ledger of the main text and all 17 tables, each twice:
for the original four models (`orig4`, automatically checked row by row against the published tables in `paper/templates/orig/`) and for
all six models (`all6`, the tables of the v2 paper). Outputs: `paper/out/`.
The activations in `data/acts/` (1.7 GB) are needed only for the overview figures of the v2 models (`sec1_overview/fig1.py`) and for re-running the theory and noise checks; without them `reproduce.sh` skips those figures and everything else runs.

To run the GPU experiments for a model (all tasks, all interventions; restartable):

```bash
python experiments/run_model.py "0 1 2 3" qwen:Qwen2.5-14B-Instruct:Qwen/Qwen2.5-14B-Instruct
```

## Paths

Every script reads and writes under one data folder, defined in `paths.py`: by default `data/` in this package, or the folder in the
environment variable `SHAPE_DATA` (useful for a scratch run that must not touch the released data). No script uses absolute paths.

## Layout

| Folder | Paper | Contents |
|---|---|---|
| `sec0_teaser/` | Figure 1 (teaser) | Ten tasks as point clouds; early layers vs. around the peak (`fig0.py`) |
| `sec1_overview/` | Figure 2 (overview) | (a) month rings; (b–e) landmark→city: shape change, handoff, onset, transfer (`fig1.py`; `FIG1A`/`FIG1M`/`FIG1T` choose model and task) |
| `sec3_events/` | §3, Figure 3 | Decision tree + examples + one-/multi-step writing (`fig2.py`); rule-threshold sweep |
| `sec4_handoff/` | §4, Figure 4 | Handoff curves and handoff-vs-peak scatter (`fig3.py`); recovery of handoff curves from the vector PDF (`parse_handoff_pdf.py`) |
| `sec5_onset/` | §5, Figure 5 | Ten-task onset grid (`fig4_grid.py`) and alternatives |
| `sec6_transfer/` | §6, Figure 6 | Ten-task transfer heatmaps (`fig5_grid.py`) and alternatives |
| `sec7_theory/` | §7 / Appendix | Blurred-input model, energy crossover (`figS_theory.py`) |
| `main_table/` | Table 1 | Distance of every boundary to the nearest peak, circular-shift null, per model (`table_main.py`; `MODELSET=orig4` for the original four) |
| `appendix/figures/` | Appendix | Handoff, strength sweep, transfer and event signals for all pairs (one column per model); decision tree |
| `appendix/tables/`, `appendix/tex/` | Appendix | Published per-pair tables (original four models) and the updated theory appendix |
| `experiments/` | all | Experiment and analysis code that produced `data/`, sorted by section (see `experiments/README.md`); `run_model.py` runs all GPU experiments for a model |
| `paper/` | all | `ledger.py` (every number of the main text, recomputed and printed next to the published value), `tables/gen_*.py` (the 17 tables), `appendix_numbers/` (numbers in the appendix prose), `templates/orig/` (published tables used as reproduction baseline), `out/` (generated) |
| `data/` | — | All inputs and outputs (see below) |

Figures for one model: set `FIGMODEL=<model>` (and `FIGM45=<model>` for `fig4_grid.py`/`fig5_grid.py`); outputs carry a suffix (`_qwen14b`, `_phi4`, …).

## Data

| Path | Written by (`experiments/`) | Content |
|---|---|---|
| `data/acts/` | `02_setup/dims.py` | Answer-position activations per layer (`*_acts.pt`). **Only the two v2 models** (1.7 GB); the original models' activations are in the first release's `acts_pack` |
| `data/group/<model>/<task>/ans/` | `00_lib/group.py` | Per-layer shape change, identity probe and patching (`per_layer.csv`), `summary.md`, `row.json` (accuracy); includes the smaller models of §7 |
| `data/dims/`, `data/lens/`, `data/steer_true/`, `data/multipos/` | `02_setup/*.py` | Shape-change variants, logit lens, centroid-difference steering, multi-position patching |
| `data/lens_full/` | `02_setup/lens_full.py` (rewritten, v2) | Full-vocabulary logit lens: `lens_p` and `a_lin` per layer (v2 models) |
| `data/events/` | `03_events/event_types.py`, `02_setup/shape_noise.py`, `shape_cv.py` | v1 signals and events, bootstrap noise floor (`noise/`), cross-validated shape change (`cv/`) |
| `data/events_v2/` | `03_events/events_v2.py`, `stats_v2.py`, `downstream_v2.py` | v2 signals and event types (all six models), handoff layer per pair; `*_orig4.csv` = the published files for the original four models |
| `data/handoff_patch/`, `data/handoff_retrieval/` | `04_handoff/handoff_patch.py` (rewritten, v2) | Input-token vs. answer-position patching (all tasks) and retrieval-task steering (v2 models) |
| `data/handoff/` | `04_handoff/steer_handoff.py` | Input-token vs. answer-position steering for the arithmetic tasks |
| `data/handoff_from_pdf.csv` | `04_handoff/build_handoff_table.py` | Handoff curves for all pairs: original models recovered from the published figure (`handoff_from_pdf_orig4.csv`), v2 models from `handoff_patch/` etc. |
| `data/handoff_all/table.csv` | `04_handoff/build_handoff_table.py` | Handoff layer per pair (replaces the lost original table) |
| `data/steer_sweep/` | `05_onset/steer_sweep.py`, `sweep_report.py` | Strength sweep per pair and logs; `report_onset.csv` = onsets per strength (model = full name) |
| `data/steer_xfer/` | `06_transfer/steer_xfer.py` and analysis scripts | Cross-layer transfer (build layer X × apply layer Y) per pair |
| `data/lap_compare/` | `05_onset/lap_compare.py`, `appendix_tables.py` | Peaks vs. logit-lens accessibility (original four models) |
| `data/theory/` | `07_theory/*.py` | Blurred-input model: `conv_layers_orig.csv` (original 21 pairs), `conv_layers_new.csv` (v2 models), merged `conv_layers.csv` / `conv_check.md`; weekday noise check |
| `data/validation/` | `04_handoff/validate_rewrite.py` | Outputs of the rewritten handoff and full-vocabulary-lens scripts on the original Qwen2.5-7B, used to validate them against the published data |
| `data/lowacc/`, `data/peak_type/`, `data/new_metrics/`, `data/weekday_add/`, `data/fill/` | see `experiments/README.md` | As in the first release |
| `data/notes/` | — | Original run notes (Chinese) |

**Not available (lost before the first release):** the original handoff-patching and full-vocabulary-lens scripts and their raw outputs for
the original four models (v2 rewrites them, see `CHANGES.md`; the published curves are kept in `handoff_from_pdf_orig4.csv` and
`events_v2/signals_v2_orig4.csv`), the input-side decomposition of App. F, the activations of the smaller models used in §7, and
`depth_vs_shape.csv`. Scripts that still refer to these (`04_handoff/regroup.py`, `verify_patch.py`, `05_onset/lap_compare.py` default
source) are kept for documentation and do not run without them.

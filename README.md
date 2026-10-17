# Where the Answer Takes Shape

**Shape-Change Peaks Mark Computational Boundaries in LLMs**

📄 [Project page](https://bellafc.github.io/shape_peak/) · [Paper (PDF)](docs/paper.pdf)


Code and data for the paper, now covering **six instruction-tuned models (59 model–task pairs)**:
Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9B-it (the original four),
Qwen2.5-14B-Instruct and Phi-4 (14B). 

Requires Python 3, numpy, pandas, scipy, scikit-learn, matplotlib, torch, transformers, huggingface_hub
(tested with Python 3.12, torch 2.8.0, transformers 5.18.0).

## Quick start
To run the GPU experiments for a model (all tasks, all interventions; restartable):

```bash
python experiments/run_model.py "0 1 2 3" qwen:Qwen2.5-14B-Instruct:Qwen/Qwen2.5-14B-Instruct
```

## Paths

Every script reads and writes under one data folder, defined in `paths.py`: by default `data/` in this package, or the folder in the
environment variable `SHAPE_DATA` (useful for a scratch run that must not touch the released data). No script uses absolute paths.

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


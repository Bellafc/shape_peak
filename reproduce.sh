#!/bin/bash
# Recompute every statistic, table and figure of the paper from data/ (CPU only, ~15 min).
# Each table generator is run twice: orig4 (the original 4 models; checked against the published table in paper/templates/orig/) and all6.
set -e
ROOT=$(cd "$(dirname "$0")" && pwd); X=$ROOT/experiments; export SHAPE_DATA=${SHAPE_DATA:-$ROOT/data}; DATA=$SHAPE_DATA
mkdir -p $ROOT/paper/out
echo "== 1. handoff summary table and curves (original models from the paper; new models from data/handoff_patch etc.)"
python $X/04_handoff/build_handoff_table.py > /dev/null
echo "== 2. event classification v2 and its statistics"
cd $DATA/events_v2 && python $X/03_events/events_v2.py && PYTHONPATH=$X/03_events python $X/03_events/stats_v2.py > stats_v2.log && python $X/03_events/downstream_v2.py > downstream_v2.log
echo "== 3. strength-sweep onsets"
cd $X/05_onset && python sweep_report.py > $DATA/steer_sweep/report.txt
echo "== 4. blurred-input model (original 21 pairs + new pairs)"
T=$(mktemp -d); cp $DATA/theory/conv_layers_orig.csv $T/conv_layers.csv; (cd $T && python $X/07_theory/conv_analyze.py > /dev/null && cp conv_check.md $DATA/theory/conv_check_orig.md)
python -c "import pandas as pd; pd.concat([pd.read_csv('$DATA/theory/conv_layers_orig.csv'), pd.read_csv('$DATA/theory/conv_layers_new.csv')]).to_csv('$DATA/theory/conv_layers.csv', index=False)"
(cd $DATA/theory && python $X/07_theory/conv_analyze.py > /dev/null); rm -rf $T
echo "== 5. Table 1"
cd $ROOT/main_table && MODELSET=orig4 python table_main.py > table_main_orig4.txt && python table_main.py > table_main_all6.txt
echo "== 6. number ledger for the main text"
python $ROOT/paper/ledger.py orig4 > $ROOT/paper/out/ledger_orig4.txt && python $ROOT/paper/ledger.py all6 > $ROOT/paper/out/ledger_all6.txt
echo "== 7. all tables (orig4 + all6) and reproduction check against the published tables"
cd $ROOT/paper/tables
for g in gen_*.py; do python $g orig4 > /dev/null 2>&1; python $g all6 > /dev/null 2>&1; done
python check_orig4.py
echo "== 8. figures"
cd $ROOT/appendix/figures && for f in figS_events_all figS_sweep_all figS_transfer_all figS_handoff_all figS_event_tree; do python $f.py > /dev/null; done
for M in Qwen2.5-14B-Instruct Phi-4-14B; do
  if ls $DATA/acts/${M}_*_acts.pt > /dev/null 2>&1; then (cd $ROOT/sec1_overview && FIGMODEL=$M FIG1A=$M FIG1M=$M FIG1T=city python fig1.py > /dev/null); else echo "   (no activations in data/acts/: skipping the overview figure for $M)"; fi
  (cd $ROOT/sec3_events && FIGMODEL=$M python fig2.py > /dev/null); (cd $ROOT/sec4_handoff && FIGMODEL=$M python fig3.py > /dev/null)
  (cd $ROOT/sec5_onset && FIGMODEL=$M FIGM45=$M python fig4_grid.py > /dev/null); (cd $ROOT/sec6_transfer && FIGMODEL=$M FIGM45=$M python fig5_grid.py > /dev/null)
done
echo "== done: tables in paper/out/, ledgers in paper/out/ledger_*.txt, figures next to their scripts"

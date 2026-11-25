#!/bin/bash
# Rebuttal experiment: alternative shape metrics on Qwen2.5-7B-Instruct, 10 tasks, half prompt set (SUB=1).
# Run from anywhere on a GPU machine:  bash experiments/rebuttal/run.sh
# Restartable: tasks whose activations already exist are skipped. Results are copied to rebuttal_results/ at the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
export HF_HOME=${HF_HOME:-/workspace/hf}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}; NAME=${NAME:-Qwen2.5-7B-Instruct}; FAM=${FAM:-qwen}
TASKS="weekday month hour letter city continent capital2hop color ctxcolor emotion"

[ -d data ] || { echo "unzipping data.zip"; python -c "import zipfile; z=zipfile.ZipFile('data.zip'); z.extractall('.', [m for m in z.namelist() if not m.startswith('__MACOSX')])"; }
pip install -q "transformers>=4.56,<5" accelerate huggingface_hub scikit-learn pandas scipy matplotlib 2>&1 | grep -v -i warning || true
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

MP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt']))")
echo "model at $MP"
mkdir -p data/rebuttal/logs
for t in $TASKS; do
  if [ -f "data/rebuttal/acts/${NAME}_${t}.pt" ]; then echo "skip $t (done)"; continue; fi
  echo "=== extract $t"
  (cd experiments/rebuttal && SUB=1 FORCE=1 python extract_acts.py "$FAM" "$t" "$MP" "$NAME") > "data/rebuttal/logs/${t}.log" 2>&1 || true
  if [ -f "data/rebuttal/acts/${NAME}_${t}.pt" ]; then grep "^done" "data/rebuttal/logs/${t}.log"
  else echo "!!! $t failed, last lines of data/rebuttal/logs/${t}.log:"; tail -15 "data/rebuttal/logs/${t}.log"; exit 1; fi
done
python experiments/rebuttal/alt_metrics.py
python experiments/rebuttal/evaluate.py "$NAME" | tee data/rebuttal/evaluate.log

mkdir -p rebuttal_results/metrics
cp data/rebuttal/report_${NAME}.md data/rebuttal/hits_${NAME}.csv data/rebuttal/peaks_${NAME}.csv data/rebuttal/curves_${NAME}.png rebuttal_results/
cp data/rebuttal/metrics/${NAME}_*.csv rebuttal_results/metrics/
echo; echo "Results in $ROOT/rebuttal_results/ (report_${NAME}.md, curves_${NAME}.png)"

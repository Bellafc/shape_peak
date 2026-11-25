#!/bin/bash
# Rebuttal experiment: alternative shape metrics, 10 tasks, half prompt set (SUB=1), for each model in MODELS.
# MODELS entries are <family>:<name>:<HF repo>; Llama uses the ungated unsloth mirror of the official Llama-3.1-8B-Instruct weights.
# Run from anywhere on a GPU machine:  bash experiments/rebuttal/run.sh
# Restartable: tasks whose activations already exist are skipped. Results are copied to rebuttal_results/ at the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
export HF_HOME=${HF_HOME:-/workspace/hf}
MODELS=${MODELS:-"qwen:Qwen2.5-7B-Instruct:Qwen/Qwen2.5-7B-Instruct llama:Llama-3.1-8B-Instruct:unsloth/Llama-3.1-8B-Instruct"}
TASKS="weekday month hour letter city continent capital2hop color ctxcolor emotion"

[ -d data ] || { echo "unzipping data.zip"; python -c "import zipfile; z=zipfile.ZipFile('data.zip'); z.extractall('.', [m for m in z.namelist() if not m.startswith('__MACOSX')])"; }
pip install -q "transformers>=4.56,<5" accelerate huggingface_hub scikit-learn pandas scipy matplotlib 2>&1 | grep -v -i warning || true
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true  # base-image builds break once pip upgrades torch; not needed here
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

for spec in $MODELS; do
IFS=: read -r FAM NAME MODEL <<< "$spec"
echo "##### $NAME"
MP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','*.jinja']))")
echo "model at $MP"
mkdir -p data/rebuttal/logs
for t in $TASKS; do
  if [ -f "data/rebuttal/acts/${NAME}_${t}.pt" ]; then echo "skip $t (done)"; continue; fi
  echo "=== extract $t"
  (cd experiments/rebuttal && SUB=1 FORCE=1 python extract_acts.py "$FAM" "$t" "$MP" "$NAME") > "data/rebuttal/logs/${NAME}_${t}.log" 2>&1 || true
  if [ -f "data/rebuttal/acts/${NAME}_${t}.pt" ]; then grep "^done" "data/rebuttal/logs/${NAME}_${t}.log"
  else echo "!!! $t failed, last lines of data/rebuttal/logs/${NAME}_${t}.log:"; tail -15 "data/rebuttal/logs/${NAME}_${t}.log"; exit 1; fi
done
if [ ! -f "rebuttal_results/report_${NAME}.md" ]; then
python experiments/rebuttal/alt_metrics.py data/rebuttal/acts/${NAME}_*.pt
python experiments/rebuttal/evaluate.py "$NAME" | tee data/rebuttal/evaluate_${NAME}.log
fi

mkdir -p rebuttal_results/metrics
[ -f "rebuttal_results/report_${NAME}.md" ] || cp data/rebuttal/report_${NAME}.md data/rebuttal/hits_${NAME}.csv data/rebuttal/peaks_${NAME}.csv data/rebuttal/curves_${NAME}.png data/rebuttal/robustness_${NAME}.md rebuttal_results/
cp data/rebuttal/metrics/${NAME}_*.csv rebuttal_results/metrics/
# pilots: multi-step arithmetic word problems and multiple-choice kinship reasoning
for pil in arith kin; do
  echo "=== pilot $pil ($NAME)"
  (cd experiments/rebuttal && python pilot.py "$FAM" "$MP" "$NAME" "$pil") > "data/rebuttal/logs/${NAME}_pilot_${pil}.log" 2>&1 \
    || { echo "!!! pilot $pil failed:"; tail -15 "data/rebuttal/logs/${NAME}_pilot_${pil}.log"; exit 1; }
  grep -E "acc=|done" "data/rebuttal/logs/${NAME}_pilot_${pil}.log" || true
done
done
python experiments/rebuttal/pilot_eval.py | tee data/rebuttal/pilot_eval.log
mkdir -p rebuttal_results/pilot
cp data/rebuttal/pilot_report.md data/rebuttal/pilot_*.png rebuttal_results/; cp data/rebuttal/pilot/*.csv rebuttal_results/pilot/
echo; echo "Results in $ROOT/rebuttal_results/"

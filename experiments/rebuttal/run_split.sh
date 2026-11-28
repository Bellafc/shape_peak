#!/bin/bash
# Rebuttal (DQWj): peaks and causal boundaries from disjoint prompt halves.
# Half A (SUB=1: offsets k = 1, 3, 5 / first half of the cues of each class) gives the shape-change peaks (forward passes only).
# Half B (SUB=2: k = 2, 4, 6 / second half of the cues) gives the shape too, and the causal boundaries, with the paper's own scripts:
#   04_handoff/handoff_patch.py (patching of the input token and the answer positions; steering for retrieval tasks)
#   04_handoff/steer_handoff.py  (steering for cyclic-arithmetic tasks)
# Outputs under data/split/{A,B} (separate from the release data). Restartable: finished steps are skipped.
# Run from anywhere on a GPU machine:  bash experiments/rebuttal/run_split.sh ; then python experiments/rebuttal/split_eval.py
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd); R=$ROOT/data/split; mkdir -p $R/A $R/B $R/logs
export HF_HOME=${HF_HOME:-/workspace/hf}
MODELS=${MODELS:-"qwen:Qwen2.5-7B-Instruct:Qwen/Qwen2.5-7B-Instruct llama:Llama-3.1-8B-Instruct:unsloth/Llama-3.1-8B-Instruct mistral:Mistral-7B-Instruct-v0.3:mistralai/Mistral-7B-Instruct-v0.3 gemma:Gemma-2-9B-it:unsloth/gemma-2-9b-it qwen:Qwen2.5-14B-Instruct:Qwen/Qwen2.5-14B-Instruct phi:Phi-4-14B:microsoft/phi-4"}
TASKS=${TASKS:-"weekday month city continent"}
ARITH=" weekday month hour letter "

pip install -q "transformers>=4.56,<5" accelerate huggingface_hub protobuf sentencepiece scikit-learn pandas scipy matplotlib 2>&1 | grep -v -i warning || true
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

step() {  # step <done-file> <log> <cmd...>: run once, keep the log, stop on failure
  local done=$1 log=$2; shift 2
  if [ -f "$done" ]; then echo "skip $(basename $log)"; return; fi
  if "$@" > "$log" 2>&1; then touch "$done"; echo "ok   $(basename $log) $(grep -E '^done|acc' "$log" | tail -1 | cut -c1-120)"
  else echo "!!! $(basename $log) failed:"; tail -20 "$log"; exit 1; fi
}

for spec in $MODELS; do
  IFS=: read -r FAM NAME MODEL <<< "$spec"
  echo "##### $NAME"
  if [ -f $R/logs/${NAME}.model.ok ] || [ $(ls $R/A/rebuttal/metrics/${NAME}_*.csv $R/B/rebuttal/metrics/${NAME}_*.csv 2>/dev/null | wc -l) -ge $(( 2 * $(wc -w <<< "$TASKS") )) ]; then
    echo "skip $NAME (all steps done)"; continue
  fi
  keep="models--${MODEL//\//--}"
  for d in "$HF_HOME"/hub/models--*; do
    if [ -e "$d" ] && [ "$(basename "$d")" != "$keep" ]; then echo "prune $(basename "$d")"; rm -rf "$d"; fi
  done
  MP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','model*.safetensors','tokenizer*','*.model','*.txt','*.jinja','*.tiktoken']))")
  export MK=$FAM MNAME_OVERRIDE=$NAME FORCE=1
  for t in $TASKS; do
    L=$R/logs/${NAME}_${t}
    step $L.actsA.ok $L.actsA.log env SUB=1 SHAPE_DATA=$R/A python experiments/rebuttal/extract_acts.py $FAM $t $MP $NAME
    step $L.actsB.ok $L.actsB.log env SUB=2 SHAPE_DATA=$R/B python experiments/rebuttal/extract_acts.py $FAM $t $MP $NAME
    step $L.patch.ok $L.patch.log env SUB=2 SHAPE_DATA=$R/B bash -c "cd experiments/04_handoff && python handoff_patch.py $MP $t $NAME"
    if [[ "$ARITH" == *" $t "* ]]; then
      step $L.steer.ok $L.steer.log env SUB=2 SHAPE_DATA=$R/B bash -c "cd experiments/04_handoff && python steer_handoff.py $MP $t $NAME"
    fi
  done
  # shape curves from the stored activations (CPU), then drop the activations to save disk
  for h in A B; do SHAPE_DATA=$R/$h python experiments/rebuttal/alt_metrics.py $R/$h/rebuttal/acts/${NAME}_*.pt > $R/logs/${NAME}_metrics_$h.log 2>&1; done
  rm -f $R/A/rebuttal/acts/${NAME}_*.pt $R/B/rebuttal/acts/${NAME}_*.pt; touch $R/logs/${NAME}.model.ok
done
SHAPE_DATA=$ROOT/data python experiments/rebuttal/split_eval.py | tee $R/split_eval.log
echo "Results in $R/split_report.md"

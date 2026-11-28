#!/bin/bash
# Rebuttal (rTuT W1): store last-token activations (forward passes only, full prompt set) so that Eq. 1 can be computed with
# prompts grouped by the input (start item) instead of the answer. Qwen2.5-7B and Llama-3.1-8B, weekday and month.
# Outputs: data/inputgroup/rebuttal/acts/<name>_<task>.pt (H, ans, src, k, t). Analysis on CPU: experiments/rebuttal/input_group.py
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd); O=$ROOT/data/inputgroup; mkdir -p $O/logs
export HF_HOME=${HF_HOME:-/workspace/hf}
MODELS=${MODELS:-"qwen:Qwen2.5-7B-Instruct:Qwen/Qwen2.5-7B-Instruct llama:Llama-3.1-8B-Instruct:unsloth/Llama-3.1-8B-Instruct"}
TASKS=${TASKS:-"weekday month"}
pip install -q "transformers>=4.56,<5" accelerate huggingface_hub protobuf sentencepiece scikit-learn pandas scipy matplotlib 2>&1 | grep -v -i warning || true
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
for spec in $MODELS; do
  IFS=: read -r FAM NAME MODEL <<< "$spec"
  echo "##### $NAME"
  MP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','model*.safetensors','tokenizer*','*.model','*.txt','*.jinja','*.tiktoken']))")
  for t in $TASKS; do
    L=$O/logs/${NAME}_${t}
    if [ -f $L.ok ]; then echo "skip $NAME $t"; continue; fi
    if MK=$FAM MNAME_OVERRIDE=$NAME SHAPE_DATA=$O python experiments/rebuttal/extract_acts.py $FAM $t $MP $NAME > $L.log 2>&1; then
      touch $L.ok; echo "ok   $NAME $t $(grep -E '^done' $L.log | tail -1)"
    else echo "!!! $NAME $t failed:"; tail -20 $L.log; exit 1; fi
  done
done
echo "Results in $O"

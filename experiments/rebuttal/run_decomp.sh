#!/bin/bash
# Rebuttal (rTuT C1): steering with the answer part vs the blurred-input part of the centroid difference (steer_decomp.py).
# Small run: Qwen2.5-7B (one-step) and Llama-3.1-8B (multi-step), weekday and month. Outputs under data/decomp/.
# Run from anywhere on a GPU machine:  bash experiments/rebuttal/run_decomp.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd); O=$ROOT/data/decomp; mkdir -p $O/logs
export HF_HOME=${HF_HOME:-/workspace/hf}
MODELS=${MODELS:-"qwen:Qwen2.5-7B-Instruct:Qwen/Qwen2.5-7B-Instruct llama:Llama-3.1-8B-Instruct:unsloth/Llama-3.1-8B-Instruct"}
TASKS=${TASKS:-"weekday month"}
pip install -q "transformers>=4.56,<5" accelerate huggingface_hub protobuf sentencepiece scikit-learn pandas scipy matplotlib 2>&1 | grep -v -i warning || true
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
for spec in $MODELS; do
  IFS=: read -r FAM NAME MODEL <<< "$spec"
  echo "##### $NAME"
  MP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','model*.safetensors','tokenizer*','*.model','*.txt','*.jinja','*.tiktoken']))")
  for t in $TASKS; do
    L=$O/logs/${NAME}_${t}
    if [ -f $L.ok ]; then echo "skip $NAME $t"; continue; fi
    if MK=$FAM MNAME_OVERRIDE=$NAME python experiments/rebuttal/steer_decomp.py $MP $t $NAME > $L.log 2>&1; then
      touch $L.ok; echo "ok   $NAME $t $(grep -E 'correct' $L.log | tail -1)"
    else echo "!!! $NAME $t failed:"; tail -20 $L.log; exit 1; fi
  done
done
echo "Results in $O"

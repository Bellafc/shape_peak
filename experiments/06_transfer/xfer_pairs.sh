#!/bin/bash
DATA=${SHAPE_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}
# CUDA_VISIBLE_DEVICES=<卡> bash xfer_pairs.sh <卡名> "mk:task mk:task ..."
L=$DATA/steer_xfer/queue_$1.log; cd "$(dirname "$0")"
for item in $2; do
  MK=${item%%:*}; T=${item##*:}
  case $MK in
    llama) REPO=unsloth/Llama-3.1-8B-Instruct; NAME=Llama-3.1-8B-Instruct; HH=/dev/shm/hf;;
    qwen) REPO=Qwen/Qwen2.5-7B-Instruct; NAME=Qwen2.5-7B-Instruct; HH=/workspace/.cache/huggingface;;
    gemma) REPO=unsloth/gemma-2-9b-it; NAME=Gemma-2-9B-it; HH=/dev/shm/hf;;
    mistral) REPO=unsloth/mistral-7b-instruct-v0.3; NAME=Mistral-7B-Instruct-v0.3; HH=/dev/shm/hf;;
  esac
  export MK HF_HOME=$HH
  P=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt']))")
  echo "=== $(date +%H:%M:%S) $NAME $T" >> $L
  python steer_xfer.py $P $T $NAME >> $L 2>&1 || echo "!!! FAILED $NAME $T" >> $L
done
echo "=== $(date +%H:%M:%S) ALL DONE" >> $L

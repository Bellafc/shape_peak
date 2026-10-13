#!/bin/bash
DATA=${SHAPE_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}
# CUDA_VISIBLE_DEVICES=<卡> bash xfer_queue.sh <MK> <repo> <名稱> <HF_HOME> "<任務...>"
MK=$1; REPO=$2; NAME=$3; export HF_HOME=$4 MK; cd "$(dirname "$0")"; L=$DATA/steer_xfer/$NAME.log
P=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt']))")
for t in $5; do echo "=== $(date +%H:%M:%S) $t" >> $L; python steer_xfer.py $P $t $NAME >> $L 2>&1 || echo "!!! FAILED $t" >> $L; done
echo "=== $(date +%H:%M:%S) ALL DONE" >> $L

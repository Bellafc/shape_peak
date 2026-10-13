#!/bin/bash
DATA=${SHAPE_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}
# 用法：CUDA_VISIBLE_DEVICES=<卡> bash fill_queue.sh <MK> <repo> <名稱> <HF_HOME> "<dims 任務>" "<group 任務>" "<FORCE group 任務>" "<steer 任務>" "<lens 任務>" "<multipos 任務>"
MK=$1; REPO=$2; NAME=$3; export HF_HOME=$4; export MK
cd "$(dirname "$0")"; L=$DATA/fill/$NAME.log
P=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt']))")
run() { echo "=== $(date +%H:%M:%S) $*" >> $L; "$@" >> $L 2>&1 || echo "!!! FAILED $*" >> $L; }
for t in $5; do run python dims.py $MK $REPO $t $NAME; done
for t in $6; do run python ../00_lib/group.py $MK $t ans $P; done
for t in $7; do export FORCE=1; run python ../00_lib/group.py $MK $t ans $P; done
unset FORCE; for t in $8; do run python steer_true.py $P $t $NAME; done
[ -n "$9" ] && run python lens.py $MK $P $NAME $9
for t in ${10}; do run python multipos.py $P $t $NAME; done
echo "=== $(date +%H:%M:%S) ALL DONE" >> $L

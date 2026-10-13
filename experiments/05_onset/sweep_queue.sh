# 強度掃描佇列（單卡）：一次一個模型，跑完刪權重。用法：bash sweep_queue.sh "llama:unsloth/Llama-3.1-8B-Instruct:Llama-3.1-8B-Instruct" ...
DATA=${SHAPE_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}
cd "$(dirname "$0")"; L=$DATA/steer_sweep/queue.log; mkdir -p $DATA/steer_sweep
TASKS=${TASKS:-"color emotion ctxcolor city continent capital2hop weekday month hour letter"}
for spec in "$@"; do IFS=: read MK REPO NAME <<< "$spec"
  M=/workspace/models/$(basename $REPO); SH=/dev/shm/models/$(basename $REPO)
  echo "=== $(date +%T) 下載 $NAME" >> $L
  python /workspace/vlm_big/clock/fetch_model.py $REPO $M $SH >> $DATA/steer_sweep/fetch_$NAME.log 2>&1 || { echo "!!! 下載失敗 $NAME" >> $L; rm -rf $M $SH; continue; }
  for t in $TASKS; do echo "=== $(date +%T) $NAME $t" >> $L
    MK=$MK python steer_sweep.py $M $t $NAME > $DATA/steer_sweep/${NAME}_$t.log 2>&1 || echo "!!! FAILED $NAME $t: $(grep -E 'Error' $DATA/steer_sweep/${NAME}_$t.log | tail -1)" >> $L
  done
  rm -rf $M $SH; echo "=== $(date +%T) 已刪除 $NAME 權重" >> $L
done
echo "=== $(date +%T) ALL DONE" >> $L

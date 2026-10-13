#!/bin/bash
DATA=${SHAPE_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}
# 強度掃描（多卡）：bash sweep_run.sh [卡數，預設全部]   輸出寫到 $DATA
# 1. 先依序下載 3 個模型（避免多卡同時下載同一個）；2. 把還沒跑完的「模型:任務」放進共享佇列（重任務排前）；3. 每張卡一個 worker 輪流領；4. 全部結束後彙總
cd "$(dirname "$0")"; O=$DATA/steer_sweep; mkdir -p $O; Q=$O/shared_queue.txt
NGPU=$(nvidia-smi -L | wc -l); NG=${1:-$NGPU}   # worker 數可多於卡數，多出的按 g % 卡數 共用卡
MODELS=${MODELS:-"gemma qwen mistral llama"}
TASKS="letter month hour weekday capital2hop city continent color ctxcolor emotion"   # 題目多的排前面，尾端比較均勻
spec() { case $1 in
  llama) echo "unsloth/Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct";;
  qwen) echo "Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct";;
  gemma) echo "unsloth/gemma-2-9b-it Gemma-2-9B-it";;
  mistral) echo "unsloth/mistral-7b-instruct-v0.3 Mistral-7B-Instruct-v0.3";;
esac; }
echo "=== $(date +%T) 下載模型：$MODELS" | tee -a $O/run.log
for mk in $MODELS; do read REPO NAME <<< "$(spec $mk)"
  python -c "from huggingface_hub import snapshot_download as s; s('$REPO', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt'])" > /dev/null 2>&1 || echo "!!! 下載失敗 $REPO" | tee -a $O/run.log
done
: > $Q
for t in $TASKS; do for mk in $MODELS; do read REPO NAME <<< "$(spec $mk)"
  grep -qs "^done" $O/${NAME}_$t.log || echo "$mk $t" >> $Q   # 已跑完（log 有 done）的跳過
done; done
echo "=== $(date +%T) 佇列 $(wc -l < $Q) 組，$NG 張卡" | tee -a $O/run.log
worker() { g=$1
  while true; do
    item=$(flock $Q.lock sh -c "head -n1 $Q; sed -i '1d' $Q"); [ -z "$item" ] && break
    read mk t <<< "$item"; read REPO NAME <<< "$(spec $mk)"
    echo "=== $(date +%T) 卡$g $NAME $t" >> $O/run.log
    CUDA_VISIBLE_DEVICES=$((g % NGPU)) MK=$mk python steer_sweep.py $REPO $t $NAME > $O/${NAME}_$t.log 2>&1 || echo "!!! FAILED 卡$g $NAME $t: $(grep -E 'Error' $O/${NAME}_$t.log | tail -1)" >> $O/run.log
  done
  echo "=== $(date +%T) 卡$g 結束" >> $O/run.log; }
for g in $(seq 0 $((NG - 1))); do worker $g & done; wait
echo "=== $(date +%T) 全部完成，彙總" | tee -a $O/run.log
python sweep_report.py > $O/report.txt 2>&1; python sweep_report.py 0.7 > $O/report_0.7.txt 2>&1
echo "=== $(date +%T) ALL DONE（$O/report.txt）" | tee -a $O/run.log

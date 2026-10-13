#!/bin/sh
DATA=${SHAPE_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}
# 依 ${CASES} 逐一跑 steer_handoff.py；換模型前清 HF 快取（第一個案例沿用現有快取）
cd "$(dirname "$0")"; prev=""
while IFS="$(printf '\t')" read mk con repo name; do
  [ -z "$mk" ] && continue
  if [ -n "$prev" ] && [ "$repo" != "$prev" ]; then rm -rf /workspace/.cache/huggingface/hub/*; fi
  prev=$repo; echo "== $name $con $(date +%T)"
  MK=$mk python steer_handoff.py $repo $con $name > $DATA/handoff/${name}_${con}.log 2>&1 || echo "FAIL $name $con"
done < ${CASES}
echo "== done $(date +%T)"

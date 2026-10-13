#!/bin/bash
DATA=${SHAPE_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}
# CUDA_VISIBLE_DEVICES=<卡> bash xfer_worker.sh <卡名> <先等這個 PID 結束>
Q=$DATA/steer_xfer/shared_queue.txt
while kill -0 $2 2>/dev/null; do sleep 20; done
while true; do
  item=$(flock $Q.lock sh -c "head -n1 $Q; sed -i '1d' $Q")
  [ -z "$item" ] && break
  bash "$(dirname "$0")/xfer_pairs.sh" $1 "$item"
done

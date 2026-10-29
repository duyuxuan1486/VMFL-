#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
: "${TASK:?Set TASK}"
: "${MODEL:?Set MODEL to a trained checkpoint}"
: "${GPUS:?Set GPUS to comma-separated free GPUs}"
: "${DATA_ROOT:=.}"
: "${OUTPUT:?Set OUTPUT to a results directory}"
mkdir -p "$OUTPUT"
IFS=, read -ra devices <<< "$GPUS"
count="${#devices[@]}"
extra=()
if [[ "$TASK" == vsp ]]; then extra=(--max-turns 100); fi
pids=()
files=()
for ((shard=0; shard<count; shard++)); do
  file="$OUTPUT/shard_${shard}.jsonl"
  files+=("$file")
  CUDA_VISIBLE_DEVICES="${devices[$shard]}" "${PYTHON:-python}" evaluate.py \
    --task "$TASK" --model "$MODEL" \
    --jsonl "$DATA_ROOT/data/$TASK/test.jsonl" \
    --output "$file" --shard-id "$shard" --num-shards "$count" "${extra[@]}" \
    > "$OUTPUT/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || { echo "An evaluation shard failed; inspect $OUTPUT/shard_*.log" >&2; exit 1; }
"${PYTHON:-python}" summarize.py "${files[@]}" --output "$OUTPUT/summary.json"

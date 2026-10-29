#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
: "${TASK:=maze_plan}"
case "$TASK" in
  maze_plan) : "${MODEL:=Qwen/Qwen2.5-VL-3B-Instruct}"; SAVE_STEPS=127 ;;
  maze_navigation|mini_behavior|vsp) : "${MODEL:=Qwen/Qwen2.5-VL-7B-Instruct}"; SAVE_STEPS=100 ;;
  *) echo "Unknown TASK: $TASK" >&2; exit 2 ;;
esac
: "${GPUS:=0,1,2,3}"
: "${OUTPUT:=outputs/sft_${TASK}}"
: "${DATA_ROOT:=.}"
: "${DEEPSPEED:=zero3}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export NPROC_PER_NODE="$(awk -F, '{print NF}' <<< "$GPUS")"
if (( NPROC_PER_NODE == 1 )); then
  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1
  : "${MASTER_PORT:=29599}"
  export MASTER_PORT
fi
if (( 16 % NPROC_PER_NODE != 0 )); then
  echo "GPU count must divide the paper's global batch size of 16" >&2
  exit 2
fi
ACCUMULATION=$((16 / NPROC_PER_NODE))
DATASET="$DATA_ROOT/data/${TASK}/train.jsonl"
test -f "$DATASET" || { echo "Missing dataset: $DATASET" >&2; exit 2; }
test -d data/images || { echo "Missing data/images; link it to the dataset image directory" >&2; exit 2; }
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"${PYTHON:-python}" -m swift.cli.main sft \
  --model "$MODEL" \
  --model_type qwen2_5_vl \
  --train_type full \
  --freeze_vit true \
  --dataset "$DATASET" \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$ACCUMULATION" \
  --learning_rate 1e-5 \
  --max_length 20480 \
  --warmup_ratio 0.05 \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 5 \
  --logging_steps 10 \
  --output_dir "$OUTPUT" \
  --dataloader_num_workers 4 \
  --gradient_checkpointing true \
  --deepspeed "$DEEPSPEED" \
  --report_to none \
  --seed 42 \
  "${resume_args[@]}"

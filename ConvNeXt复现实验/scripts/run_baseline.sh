#!/usr/bin/env bash
# 根据当时日志中的 Namespace 参数整理；适用于已作兼容修改的官方代码。
set -euo pipefail

model="${1:?用法: bash scripts/run_baseline.sh convnext_tiny|resnet50}"
case "$model" in convnext_tiny|resnet50) ;; *) echo "不支持的模型: $model" >&2; exit 2 ;; esac

REPO="${REPO:-/home/jovyan/work/ConvNeXt}"
PYTHON="${PYTHON:-/home/jovyan/work/convnext_env/bin/python}"
TRAIN_DIR="${TRAIN_DIR:-/tmp/imagenet100/imagefolder/train}"
VAL_DIR="${VAL_DIR:-/tmp/imagenet100/imagefolder/val}"
OUT_ROOT="${OUT_ROOT:-/home/jovyan/work/experiments}"

[[ -f "$REPO/main.py" ]] || { echo "找不到 $REPO/main.py" >&2; exit 1; }
[[ -d "$TRAIN_DIR" && -d "$VAL_DIR" ]] || { echo "找不到 ImageFolder 数据目录" >&2; exit 1; }

if [[ "$model" == convnext_tiny ]]; then
  out="$OUT_ROOT/convnext_tiny_imagenet100"
else
  out="$OUT_ROOT/resnet50_imagenet100_126k"
fi
mkdir -p "$out"
cd "$REPO"

"$PYTHON" -u main.py \
  --model "$model" --batch_size 32 --update_freq 16 --epochs 100 \
  --input_size 224 --nb_classes 100 --drop_path 0.1 \
  --opt adamw --lr 0.0005 --min_lr 0.000001 --warmup_epochs 5 \
  --weight_decay 0.05 --layer_decay 1.0 \
  --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 --reprob 0.25 \
  --model_ema true --model_ema_decay 0.9999 --model_ema_eval true \
  --data_set image_folder --data_path "$TRAIN_DIR" --eval_data_path "$VAL_DIR" \
  --output_dir "$out" --log_dir "$out/tensorboard" \
  --num_workers 0 --seed 42 --use_amp true \
  2>&1 | tee "$out/train_console.log"

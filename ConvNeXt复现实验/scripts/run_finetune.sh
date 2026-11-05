#!/usr/bin/env bash
# 从自己训练的 ConvNeXt-Tiny 最佳权重开始，重新进行 10 轮微调。
# 用法: bash scripts/run_finetune.sh lr_only|no_mix
# 原版 main.py 对较新 PyTorch 的检查点加载方式需要兼容；见下方提示。
set -euo pipefail

mode="${1:?用法: bash scripts/run_finetune.sh lr_only|no_mix}"
case "$mode" in lr_only|no_mix) ;; *) echo "不支持的模式: $mode" >&2; exit 2 ;; esac

REPO="${REPO:-/home/jovyan/work/ConvNeXt}"
PYTHON="${PYTHON:-/home/jovyan/work/convnext_env/bin/python}"
TRAIN_DIR="${TRAIN_DIR:-/tmp/imagenet100/imagefolder/train}"
VAL_DIR="${VAL_DIR:-/tmp/imagenet100/imagefolder/val}"
OUT_ROOT="${OUT_ROOT:-/home/jovyan/work/experiments}"
BASE_CKPT="${BASE_CKPT:-$OUT_ROOT/convnext_tiny_imagenet100/checkpoint-best.pth}"

[[ -f "$REPO/main.py" && -f "$BASE_CKPT" ]] || { echo "找不到源码或原始最佳检查点" >&2; exit 1; }
[[ -d "$TRAIN_DIR" && -d "$VAL_DIR" ]] || { echo "找不到 ImageFolder 数据目录" >&2; exit 1; }

if [[ "$mode" == no_mix ]]; then
  out="$OUT_ROOT/convnext_tiny_imagenet100_no_mix_10ep"
  mixup=0
  cutmix=0
else
  out="$OUT_ROOT/convnext_tiny_imagenet100_finetune_lr1e5_10ep"
  mixup=0.8
  cutmix=1.0
fi
mkdir -p "$out"

# 自己训练的旧格式检查点可先转换成只含模型参数的文件。
weights="$out/finetune_input_weights.pth"
"$PYTHON" "$(dirname "$0")/prepare_finetune_checkpoint.py" "$BASE_CKPT" "$weights"
cd "$REPO"

"$PYTHON" -u main.py \
  --model convnext_tiny --batch_size 32 --update_freq 16 --epochs 10 \
  --input_size 224 --nb_classes 100 --drop_path 0.1 \
  --opt adamw --lr 0.00001 --min_lr 0.000001 --warmup_epochs 0 \
  --weight_decay 0.05 --layer_decay 1.0 \
  --mixup "$mixup" --cutmix "$cutmix" --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 --reprob 0.25 \
  --model_ema true --model_ema_decay 0.9999 --model_ema_eval true \
  --finetune "$weights" \
  --data_set image_folder --data_path "$TRAIN_DIR" --eval_data_path "$VAL_DIR" \
  --output_dir "$out" --log_dir "$out/tensorboard" \
  --num_workers 0 --seed 42 --use_amp true \
  2>&1 | tee "$out/train_console.log"

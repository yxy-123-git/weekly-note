"""对 ImageFolder 验证集做独立 FP32 Top-1 / Top-5 评估。

示例：
python scripts/validate_fp32.py --repo /home/jovyan/work/ConvNeXt \
  --model convnext_tiny --checkpoint /path/to/checkpoint-best.pth \
  --val-dir /tmp/imagenet100/imagefolder/val

本脚本使用普通模型权重 `model`，不读取 EMA 权重。检查点必须来自可信来源。
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", choices=["convnext_tiny", "resnet50"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("需要 CUDA GPU")
    sys.path.insert(0, str(args.repo.resolve()))
    import models  # noqa: F401  注册官方 ConvNeXt 模型
    import timm

    torch.set_num_threads(4)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = datasets.ImageFolder(args.val_dir, transform=transform)
    if len(dataset) != 5000 or len(dataset.classes) != 100:
        raise ValueError(f"预期 5000 张、100 类，实际 {len(dataset)} 张、{len(dataset.classes)} 类")

    # 用户自己训练的旧格式检查点可能包含 NumPy 元数据；仅加载可信文件。
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model")
    if state is None:
        raise KeyError("找不到 'model' 权重；本脚本不评估 EMA")
    model = timm.create_model(args.model, pretrained=False, num_classes=100)
    model.load_state_dict(state, strict=True)
    model = model.float().cuda().eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    top1 = top5 = total = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.cuda(non_blocking=True).float()
            targets = targets.cuda(non_blocking=True)
            predictions = model(images).topk(5, dim=1).indices
            top1 += (predictions[:, 0] == targets).sum().item()
            top5 += (predictions == targets[:, None]).any(dim=1).sum().item()
            total += targets.numel()
            if total % 320 == 0 or total == len(dataset):
                print(f"已验证 {total}/{len(dataset)} 张", flush=True)

    result = {"model": args.model, "checkpoint": str(args.checkpoint),
              "images": total, "precision": "FP32", "top1_percent": 100 * top1 / total,
              "top5_percent": 100 * top5 / total}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

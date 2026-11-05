"""从两个实验目录的 log.txt 绘制训练损失与验证 Top-1 曲线。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if all(k in r for k in ("epoch", "train_loss", "test_acc1"))]
    if not rows:
        raise ValueError(f"日志中没有完整的轮次/损失/验证记录: {path}")
    return sorted(rows, key=lambda r: r["epoch"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--convnext-log", type=Path, required=True)
    parser.add_argument("--resnet-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {"ConvNeXt-Tiny": read_log(args.convnext_log),
            "ResNet-50": read_log(args.resnet_log)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, rows in runs.items():
        epoch = [row["epoch"] + 1 for row in rows]
        axes[0].plot(epoch, [row["train_loss"] for row in rows], label=name)
        axes[1].plot(epoch, [row["test_acc1"] for row in rows], label=name)
        best = max(rows, key=lambda row: row["test_acc1"])
        print(f"{name}: 第 {best['epoch'] + 1} 轮，日志 Top-1 {best['test_acc1']:.2f}%，Top-5 {best.get('test_acc5', float('nan')):.2f}%")
    for ax, title, ylabel in zip(axes, ("Training loss", "Validation Top-1"), ("Loss", "Accuracy (%)")):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"曲线已保存: {args.output}")


if __name__ == "__main__":
    main()

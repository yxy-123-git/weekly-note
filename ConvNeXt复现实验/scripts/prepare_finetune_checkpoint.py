"""从自己训练的检查点提取模型权重，供官方 main.py 的 --finetune 使用。

只可对来源可信、自己训练得到的检查点运行。旧检查点含 NumPy 对象，
在新版本 PyTorch 中使用 weights_only=False 加载会执行 pickle 反序列化。
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("输入与输出路径不能相同")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError("检查点中找不到普通模型的 'model' 权重")
    state = checkpoint["model"]
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise TypeError("'model' 不是纯张量 state_dict")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, args.destination)
    print(f"已写入: {args.destination}")


if __name__ == "__main__":
    main()

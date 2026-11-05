# ConvNeXt 图像分类复现：ImageNet-100

这组实验以 [ConvNeXt 官方实现](https://github.com/facebookresearch/ConvNeXt)为基础，在 ImageNet-100 上比较 ConvNeXt-Tiny 和 ResNet-50，并尝试对 ConvNeXt-Tiny 做短程微调。训练数据、日志和模型权重保存在远程 JupyterLab 工作空间；本目录归档实验记录、汇总结果和复跑所需的脚本。

## 主要结果

两种模型均在 126,689 张训练图上从随机初始化训练 100 轮，使用同一份 5,000 张验证集。下表是重新加载各自最佳检查点后的独立 FP32 验证结果。

| 模型 | 最佳轮次 | Top-1 | Top-5 |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 97 | 82.78% | 95.90% |
| ResNet-50 | 96 | 81.64% | 95.18% |
| ConvNeXt-Tiny：再微调 10 轮，关闭 Mixup/CutMix | 见微调日志 | 83.22% | 96.06% |

主实验中，ConvNeXt-Tiny 的 Top-1 比 ResNet-50 高 1.14 个百分点。微调后的 Top-1 比原 ConvNeXt-Tiny 检查点高 0.44 个百分点。小样本实验的排序不同：每类只用 200 张图训练时，ResNet-50 的最佳 Top-1 为 49.84%，ConvNeXt-Tiny 为 43.02%。因此这里讨论的是**本次数据与训练设置下的结果**，不能拿 83.22% 直接与原论文 ImageNet-1K 上的准确率比较。

## 文件说明

| 路径 | 内容 |
|---|---|
| [docs/复现报告.md](docs/复现报告.md) | 数据、环境、训练设置、结果和问题记录 |
| [docs/原始产物清单.md](docs/原始产物清单.md) | 远程日志、检查点和数据清单的位置及归档状态 |
| [results/实验结果.csv](results/实验结果.csv) | 本次实验的汇总数值，标明日志值与独立验证值 |
| [scripts/run_baseline.sh](scripts/run_baseline.sh) | 根据日志参数整理的主实验启动脚本 |
| [scripts/run_finetune.sh](scripts/run_finetune.sh) | 两种 10 轮微调设置的启动脚本 |
| [scripts/prepare_finetune_checkpoint.py](scripts/prepare_finetune_checkpoint.py) | 将自己训练的旧格式检查点转为仅含模型权重的文件 |
| [scripts/validate_fp32.py](scripts/validate_fp32.py) | 用最佳检查点对全部验证集做 FP32 Top-1/Top-5 复核 |
| [scripts/plot_curves.py](scripts/plot_curves.py) | 从官方训练日志生成训练损失和验证 Top-1 曲线 |
| [scripts/plot_efficiency.py](scripts/plot_efficiency.py) | 用记录的吞吐与准确率绘制辅助图 |
| [figures/accuracy_throughput_fp32.png](figures/accuracy_throughput_fp32.png) | FP32 准确率与推理吞吐对照图 |

## 如何复跑

训练依赖官方 ConvNeXt 代码，以及本次环境中的兼容性修改。原始代码版本记录为提交 `048efcea897d999aed302f2639b6270aedf8d4c8`。先准备 ImageFolder 格式的训练集和验证集，并设置 `REPO`、`PYTHON`、`TRAIN_DIR`、`VAL_DIR`、`OUT_ROOT`。之后在 Linux 工作空间执行：

```bash
bash scripts/run_baseline.sh convnext_tiny
bash scripts/run_baseline.sh resnet50
```

单卡下，主实验的单步批量为 32、梯度累积 16，有效批量为 512。脚本根据实验记录重建，**不是当时 JupyterLab 启动单元的逐字导出**。尤其是官方旧代码与当前 PyTorch、timm 的兼容改动尚未收进本仓库；在未经适配的新环境里，直接运行可能报错。实际复跑前请核对 `main.py --help`、环境版本和检查点路径。微调及验证命令见各脚本开头的说明。

本仓库不包含 ImageNet 图像、模型权重或完整训练日志。远程原件位置见[原始产物清单](docs/原始产物清单.md)。

## 参考

- Liu et al., [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html), CVPR 2022。
- [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)。

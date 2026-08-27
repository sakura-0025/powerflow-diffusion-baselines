# 输电网潮流扩散Baseline

本目录提供三种统一接口的基线：普通DDPM、Wang等的物理信息DDPM、Hoseinpour和Dvorkin的约束扩散。项目使用`uv`管理，并包含IEEE 14、30、118三个标准算例的AC-OPF数据构建代码。

## 目录

```text
baseline/
├── data/                 # 三个标准算例的数据生成与加载
├── model/                # DDPM、Wang、Hoseinpour模型与AC物理约束
├── evaluation/           # 论文指标、统一补充指标和三模型可视化
├── docs/                 # 论文协议、数据清单和复现状态
├── tests/                # 公式与张量形状机制测试
├── train.py              # 统一训练入口
├── sample.py             # 统一采样入口
└── evaluate.py           # 统一统计、物理和速度评估
```

论文公式、实现位置与推断项的逐项对应见`docs/PAPER_EVIDENCE.md`；机器可读协议见`docs/protocol.json`。

## 环境

```bash
cd baseline
uv sync --extra dev
uv run pytest -q
```

## 构建三个统一比较数据集

下面的`common`协议对三种模型使用完全相同的AC-OPF样本。正式样本数可根据计算预算调整。

```bash
uv run baseline-build --case ieee14  --samples 150000 --protocol common --seed 2026 --output datasets/ieee14_common.npz
uv run baseline-build --case ieee30  --samples 150000 --protocol common --seed 2026 --output datasets/ieee30_common.npz
uv run baseline-build --case ieee118 --samples 150000 --protocol common --seed 2026 --output datasets/ieee118_common.npz
```

每个数据文件生成后先审计；若形状、划分、有限值或交流潮流残差不合格，命令会返回失败状态：

```bash
uv run baseline-audit --data datasets/ieee14_common.npz --output outputs/audit_ieee14.json
```

论文特定协议：

- `--protocol wang`：负荷为额定值的80%–120%，发电成本系数为50%–150%；
- `--protocol hoseinpour`：负荷为额定值的80%–100%；
- `--protocol common`：80%–120%负荷，不扰动成本，用于公平模型对比。

Wang原文使用IEEE 14/30，Hoseinpour原文使用PJM 5/IEEE 24/IEEE 118。这里选择IEEE 14/30/118，是因为它们都属于两篇论文实际使用过的标准算例，同时覆盖小、中、大三种规模。三种模型在正式横向对比时统一使用上述`common`数据，论文特定协议只用于复现敏感性实验。

## 训练

普通DDPM：

```bash
uv run baseline-train --method ddpm --data datasets/ieee14_common.npz --output-dir outputs/ieee14_ddpm --epochs 100 --steps 200 --seed 2026
```

Wang PI-DDPM：

```bash
uv run baseline-train --method wang --data datasets/ieee14_common.npz --output-dir outputs/ieee14_wang --epochs 100 --steps 200 --schedule-epochs 20 --physics-weight 1 --seed 2026
```

Hoseinpour约束扩散的训练阶段只训练两个解耦去噪器：

```bash
uv run baseline-train --method hoseinpour --data datasets/ieee14_common.npz --output-dir outputs/ieee14_hoseinpour --epochs 100 --steps 200 --seed 2026
```

## 采样

```bash
uv run baseline-sample --data datasets/ieee14_common.npz --checkpoint outputs/ieee14_ddpm/best.pt --num-samples 10000 --output outputs/ieee14_ddpm/generated.npz
uv run baseline-sample --data datasets/ieee14_common.npz --checkpoint outputs/ieee14_wang/best.pt --num-samples 10000 --output outputs/ieee14_wang/generated.npz
uv run baseline-sample --data datasets/ieee14_common.npz --checkpoint outputs/ieee14_hoseinpour/best.pt --guidance-scale 5e-4 --num-samples 10000 --output outputs/ieee14_hoseinpour/generated.npz
```

Hoseinpour论文报告的引导系数随系统不同：PJM 5使用`1e-2`，IEEE 24使用`1e-4`，IEEE 118使用`5e-4`。IEEE 14和30没有原文值，必须在验证集选择并注明为项目设置。

## 统一评估

```bash
uv run baseline-evaluate --data datasets/ieee14_common.npz --generated outputs/ieee14_ddpm/generated.npz --output outputs/ieee14_ddpm/metrics.json --split test
```

评估器将指标来源分为两组：

- 论文明确指标：Wang等式(7)的平均复功率不平衡、Hoseinpour--Dvorkin等式(33)的联合一阶Wasserstein距离，以及逐母线有功/无功失配均值和标准差；
- 项目统一补充指标：逐通道边际Wasserstein距离、MMD、相关矩阵误差、残差分位数、约束违反率、采样时间、NFE和物理梯度次数。

联合Wasserstein距离使用训练集归一化后的非恒定特征，并在固定随机子样本上求等质量最优匹配；输出会明确记录子样本数和估计方法。调节Hoseinpour引导系数时必须使用`--split validation`，最终配置确定后才能使用`--split test`。

## 三模型汇总和可视化

每个运行目录至少应包含`generated.npz`，若同时包含`training.json`，汇总表还会报告参数量、训练时间和训练曲线：

```bash
uv run baseline-compare \
  --data datasets/ieee14_common.npz \
  --run DDPM=outputs/ieee14_ddpm \
  --run Wang=outputs/ieee14_wang \
  --run Hoseinpour=outputs/ieee14_hoseinpour \
  --output-dir outputs/comparison_ieee14 \
  --split test \
  --metric-samples 1000 \
  --transport-samples 256 \
  --plot-samples 2000 \
  --seed 2026
```

输出包括：

```text
outputs/comparison_ieee14/
├── comparison.json      # 完整机器可读汇总和指标来源
├── comparison.csv       # 便于制表
├── comparison.md        # 可直接查看的指标表与图索引
├── metrics/             # 三个模型在同一评价器下重新计算的指标
└── figures/
    ├── overview.png/pdf                  # 统计、物理、约束、速度总览
    ├── marginal_distributions.png/pdf    # 四类变量边际分布
    ├── joint_distributions.png/pdf       # P-Q与V-theta二维联合分布
    ├── power_mismatch_by_bus.png/pdf     # 逐母线失配均值±标准差
    ├── residual_cdf.png/pdf              # 物理残差经验CDF
    └── training_convergence.png/pdf      # 相对训练/验证收敛曲线
```

所有图同时输出300 DPI PNG和矢量PDF，使用色盲友好颜色以及不同线型/标记。单随机种子结果不会绘制虚假的误差条，也不会计算带主观权重的“总分”；正式三随机种子完成后再报告均值和标准差。

## 论文复现边界

Wang论文同时写明“噪声预测”与“输出Sigmoid”，两者存在技术冲突。默认使用与其Eq. (5)一致的线性噪声头；`--wang-sigmoid-output`可进行严格文字设置的敏感性实验。Wang学习计划的单调化方法和Hoseinpour的网络结构没有公开，当前实现均隔离为可替换模块并记录在`docs/protocol.json`。

在完整数据、三随机种子训练、采样和协议覆盖检查完成前，本项目只能称为**部分复现/机制检查框架**。

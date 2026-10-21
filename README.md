# 酿酒酵母细胞周期神经算子动力学建模与主动学习系统 (Budding Yeast Neural Operators & Active Learning)

> **2026-10-21 模型清理**：baseline 只保留 18 个标准实现，入口为 `mlp/resnet/unet/kan/transformer/mamba/deeponet/fno/wno/mwt/gnot/transolver/lno/neural_ode/sc_fno/bregman_fno/lamo/konode`。
> 旧改造版已删除；标准入口没有额外参数调制/生物图模块。旧实验表是历史实现的结果，不能作为当前标准模型的论文结果。
> 当前来源、参数量和 SC-FNO 监督要求见 [标准基线说明](docs/BASELINE_SELECTION_AUDIT.md)。


[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0+-792EE5.svg)](https://lightning.ai)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

本项目构建了面向**酿酒酵母 (*Saccharomyces cerevisiae*) 细胞周期高度非线性生化反应系统（39 维时序变量、141 维动力学参数）**的高精度神经算子（Neural Operator）替代模型库，并配备了面向偏微分/常微分动力学系统的 **AL4PDE 百万级主动学习采集算法库**与生化物理约束（PINO ODE 残差、事件跳跃掩码、GradNorm 梯度平衡）。

---

## 📚 权威技术文档导航 (Authoritative Documentation)

完整技术体系与架构细节请参阅 `docs/` 目录下的 5 篇权威专题文档：

| 专题文档 | 核心内容概要 |
| :--- | :--- |
| 📖 [**1. 架构总览与算子模型库**](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/1_ARCHITECTURE_AND_MODELS.md) | `BioRefinerPlus` 物理细化算子结构、70+ 模型全景注册表、三代技术演进路线 |
| 🧪 [**2. 主动学习指南 (AL4PDE)**](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/2_ACTIVE_LEARNING_GUIDE.md) | 100 万未仿真候选池机制、AL4PDE 10 种生产级采集策略、分岔熵与物理残差采样 |
| ⚖️ [**3. 损失函数库与生物物理约束**](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md) | 17 种损失函数、39 通道单次 Autograd GradNorm 自适应平衡、配点法 ODE 物理残差 |
| 📊 [**4. 数据集全景与归一化规范**](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/4_DATASETS_AND_NORMALIZATION.md) | 10万级数据生成器、多关系生化拓扑图、统一 $\text{asinh}(Z)$ 模长有界归一化审计 |
| 🏆 [**5. 模型评测榜单与 Badcase 剖析**](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/5_BENCHMARK_AND_LEADERBOARD.md) | 126 个真实生物突变体金标准完整榜单、多种子稳定性、过表达/多重敲除坏 Case 归因 |
| 🤖 [**AI Agent 规范 (AGENTS.md)**](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/AGENTS.md) | AI 编码与代码落盘字典（严禁在根目录乱建临时脚本，严格归档规整） |

---

## 🌟 核心特性与架构亮点

1. **SOTA 生物物理细化神经算子 (`BioRefinerPlus`)**：
   - 在 126 个真实生物突变体金标准上取得 **Rel $L_2 = 0.1198$**、**Pearson Corr $= 0.9825$**、**MAE $= 0.0567$**，显著超越经典 FNO、Transolver、DeepONet 与 Neural ODE。
2. **AL4PDE 百万级池化主动学习 (Pool-based Active Learning)**：
   - 杜绝传统查询合成（Query Synthesis）的数据泄漏问题，从 100 万未仿真参数池中，利用模型前向推理（Monte Carlo Dropout、相空间时序形状多样性、生化通量残差、分岔状态熵）动态挑选最具信息量的样本进行 ODE 仿真增量标注。
3. **39 通道解耦 GradNorm 自适应微观梯度平衡**：
   - 独创 **单次 Autograd 求导技巧 (Single-Call Decoupled Norm Trick)**，无需多次反向传播图遍历，单步 `<2ms` 动态均衡 39 个生化变量的梯度贡献。
4. **统一双曲反正弦标准化 (`asinh(Z)`)**：
   - 将 1500 倍强启动子过表达平滑压缩至 $+3.85$，同时将完全敲除点限定在 $-1.80$，彻底根治多基因敲除向量模长爆炸引发的假分岔阻滞。

---

## 📁 规范项目目录结构

```
budding_yeast_v2/
├── AGENTS.md                  # 🤖 AI Agent 严格编码行为准则与代码落盘字典
├── README.md                  # 📘 项目总览与导航
├── train.py                   # 🚀 统一训练入口 (支持全量模型、17 种损失与 AL4PDE 主动学习)
├── eval.py                    # 📈 统一评估与可视化入口 (126 突变体金标准)
├── tune.py                    # 🔍 自动化超参搜索入口
├── run_pipeline.sh            # ⚙️ 批量基准训练与多算子对比流水线
├── run_al_pure_random_1m.sh   # 🔬 AL4PDE 百万候选池主动学习执行流水线
├── docs/                      # 📚 5 篇核心技术专题规范文档
├── models/                    # 🧠 神经网络模型库 (ours: BioRefinerPlus 等; baselines: 18 个标准实现)
├── data/                      # 💾 数据集生成脚本、生化多关系图、可达性分析
│   ├── reachability/          # 🎯 参数与相空间轨迹可达性分析报告
│   └── dataset_analysis/      # 📑 数据集分布消融与统计报告
├── utils/                     # 🛠️ 损失函数库、物理方程、AL4PDE 回调与评测指标
├── scripts/                   # 📊 规范化分析与评估工具集
│   ├── analysis/              # 🔍 Badcase 挖掘、多种子统计、动力学相图分析
│   └── eval/                  # 📐 动态时序评估与可解释性分析
├── experiments/               # 🧪 实验输出与评估日志 (按规范结构组织)
│   └── summary_tables/        # 📑 全量模型基准对比 CSV 汇总表
├── scratch/                   # 📝 临时排查探测脚本与沙盒环境 (自动 gitignore)
│   └── probes/                # 🔬 临时探测与调试脚本
└── archive/                   # 📦 历史过时脚本与旧文档归档备份
```

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备
```bash
conda activate neuralop
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
```

### 2. 标准基准训练 (Standard Benchmark Training)
```bash
python train.py \
  --model biom_refiner_plus \
  --loss hybrid_mse_rel \
  --lambda_rel 0.2 \
  --modes 24 \
  --width 192 \
  --batch_size 256 \
  --lr 8.8e-4 \
  --max_epochs 100 \
  --seed 42
```

### 3. AL4PDE 主动学习训练 (Active Learning with 1M Candidate Pool)
```bash
python train.py \
  --model biom_refiner_plus \
  --loss hybrid_mse_rel \
  --al_strategy al4pde_bifurcation_entropy \
  --pool_path /home/users/zsy/mydata/budding_yeast/pool_1m_pure_random.npz \
  --al_budget_per_round 500 \
  --al_candidate_pool_size 100000 \
  --al_warmup_epochs 8 \
  --al_interval_epochs 5 \
  --al_max_rounds 6 \
  --max_epochs 100
```

### 4. 真实突变体评估与可视化 (Evaluation)
```bash
python eval.py \
  --exp_dir experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/triple_dynamic_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42
```

---

## 🏆 核心天梯榜概览 (Leaderboard Summary)

| 排名 | 模型 | 突变体 Rel $L_2$ $\downarrow$ | Pearson Corr $\uparrow$ | MAE $\downarrow$ |
| :---: | :--- | :---: | :---: | :---: |
| 🥇 **1** | **`biom_refiner_plus` (Ours)** | **0.1198** | **0.9825** | **0.0567** |
| 🥈 **2** | **`bregman_wno`** | 0.1310 | 0.9821 | 0.0912 |
| 🥉 **3** | **`biom_refiner_event_hybrid`** | 0.1351 | 0.9785 | 0.0708 |
| 4 | **`konode`** | 0.1526 | 0.9737 | 0.1123 |
| 5 | **`icon`** | 0.1528 | 0.9743 | 0.1188 |
| 6 | **`unet1d`** | 0.1549 | 0.9726 | 0.0964 |
| 7 | **`dpot1d`** | 0.1559 | 0.9732 | 0.1072 |
| 8 | **`mwt1d`** | 0.1573 | 0.9721 | 0.1096 |
| 9 | **`gnot`** | 0.1586 | 0.9727 | 0.1130 |
| 10 | **`cno1d`** | 0.1621 | 0.9699 | 0.1235 |

*更多详细指标及消融数据，请查阅 [docs/5_BENCHMARK_AND_LEADERBOARD.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/5_BENCHMARK_AND_LEADERBOARD.md)。*

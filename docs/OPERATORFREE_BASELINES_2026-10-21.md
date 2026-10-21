# KAN、Transformer、Mamba 的初值问题适配（2026-10-21）

用户采纳后新增 `kan`、`transformer`、`mamba`，当前选定 baseline 共 18 个。本轮不增加 GINO，不修改论文原始仓库。

## 输入输出与信息边界

模型统一通过 `forward_ic_time(X,P)` 调用：

- `P:(B,D)`：全部 ODE 参数，出芽酵母 D=141。
- `X:(B,V,2,T)`：特征 0 的首个时间位置提供全部变量初值；特征 1 为查询时间。出芽酵母 V=39，通常 T=500。
- 返回 `(B,V,T)`：所有变量的完整预测轨迹。

每个查询点的输入为 `[y0_1,...,y0_V,P_1,...,P_D,t]`。初值与参数广播至所有时间点，模型不输入观测历史或未来轨迹，不把变量拆成独立 batch。数值预处理沿用 DataModule 的 train-only 统计量。

查询时间应严格递增且各变量共享；三个新增模型允许非均匀查询。KAN 是逐点条件回归，给定同一时刻的输出不依赖其他查询点；Transformer 和 Mamba 是序列模型，改变查询序列可能改变输出，不将它们宣称为完全网格无关的连续解算子。

## 具体实现与默认配置

| 入口 | 来源及保留机制 | width=64、n_blocks=4 参数量 |
| --- | --- | --- |
| `kan` | efficient-kan，B-spline 边函数、SiLU base 分支、可学习 spline scaler；隐藏宽度重复 n_blocks 次 | 263,680 |
| `transformer` | 师兄 TransformerPerTokenRegressor；4-head attention、FFN、pre-RMSNorm、正弦位置编码、输入/输出 MLP；encoder 深度 n_blocks | 171,687 |
| `mamba` | 师兄 MambaPerTokenRegressor；官方 Mamba-1 的输入依赖 dt/B/C、因果 depthwise Conv1d、A/D、SiLU 门控；保留师兄 RMSNorm、残差和 FFN | 242,343 |

统计 V=39、D=141。Transformer/Mamba 的 FFN 隐藏宽度为 `2*width`，dropout 明确设为 0；不是增加条件调制。Transformer width 必须是 4 的倍数。Mamba 固定 `d_state=16,d_conv=4,expand=2,dt_rank=ceil(width/16)`。模型宽度/深度通过已有 CLI 传递并进入 checkpoint，无新增 ref/pure/1d 别名。

KAN 保留官方 `grid_size=5,spline_order=3,grid_range=[-1,1]` 默认值。当前使用固定网格，不调用 update_grid，不自动加入 KAN 正则项；因此论文中应注明是 efficient-kan 的任务适配，不能声称逐项复现 pykan 的网格更新/正则训练协议。Z-score 输入可能超出样条覆盖区间，不能据此保证全参数范围的样条表达能力；base 分支仍参与输出。

Transformer 保留师兄 RMSNorm/MLP 具体回归骨干，不声称是未经修改的原始 Transformer 论文代码。其 attention 不加因果 mask，所有 token 都仅包含已知条件和查询时间，无未来真值。Mamba 采用因果时间顺序；此处不额外增加双向块。

没有给这三个 baseline 添加 FiLM、图先验或额外跨注意力。

## Mamba 数值后端

当前环境没有 mamba_ssm / causal_conv1d。为使模型可以实际训练和评测，保留固定版本官方 Mamba-1 初始化及 selective_scan_ref，完整序列 forward 按官方未融合路径计算。einops 排列替换为等价 Torch 操作，不用普通卷积、GRU 或固定衰减 SSM 替代输入依赖选择性计算。

CPU 使用官方参考 scan；CUDA 如可导入官方 selective_scan_fn 则调用该算子，否则使用相同参考 scan 并明确提示。当前环境的 CUDA 检查使用参考路径；未验证已安装编译扩展的快速路径。未保留 streaming inference cache，也不调用 fused mamba_inner_fn。它们不影响当前整段轨迹任务的数学结构，但参考路径的速度和显存成本不能等同于官方融合内核。正式大规模实验应安装兼容的 mamba-ssm 并核实后端成本。

固定来源：

- efficient-kan commit `7b6ce1c87f18c8bc90c208f6b494042344216b11`，MIT。
- state-spaces/mamba commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`，Apache-2.0。
- 师兄 `Tau_BNO-main/models/OperatorFree/{TransformerBackbone,MambaBackbone,NN,layers}.py`，原 MIT 文本保留。

来源 URL、原文件 SHA256 在 `models/baselines/_upstream/operatorfree_sources.json`；本地包装、数值核心和师兄源文件 SHA256 由 `scripts/analysis/audit_baseline_alignment.py` 导出。

## 训练、调参与评测

```bash
export PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle:$PYTHONPATH
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
python train.py --model kan --dataset_name /absolute/path/to/data.npz \
  --width 64 --n_blocks 4 --loss_type hybrid_mse_rel --al_strategy none
# 同样参数将 kan 替换为 transformer 或 mamba。
MODEL=transformer bash run_tune.sh
python eval.py --ckpt /absolute/path/to/best.ckpt \
  --dataset_name /absolute/path/to/data.npz --save_dir /absolute/path/to/eval_output
```

HPO 的三个新增模型均搜索 width=32/64/128、n_blocks=2..6、batch=16/32/64；不搜索其不使用的 Fourier modes、小波类型或小波层数。三者沿用现有学习率、权重衰减等训练参数。实际大规模显存/吞吐需按设备验证；HPO 的既有真实突变体调参目标没有在此改变，独立测试协议限制仍见 CLI 核查文档。

## 验证

- 全部 18 个 baseline：实际训练 parser/factory → DataModule → Lightning CPU 单步优化 → 保存 checkpoint → 标准 eval 加载与预测一致；实际 HPO 构造器与 Bash 导出/解析传参。
- 新增三者：完整初值/参数梯度、39变量/141参数/500点形状、严格 state_dict、输入信息边界、可变长度和非均匀时间、CPU/CUDA BF16 前反向。
- KAN 与固定版本 efficient-kan 源码输出一致；Transformer 与师兄原类同权重输出及参数输入梯度一致。
- Mamba 完整 mixer 与官方未融合类同权重前向及全部参数梯度一致；参考 scan 与独立稠密 Green 函数求和的前向/梯度一致；因果前缀一致，dt/B/C 投影随输入变化。

执行：

```bash
python -m pytest tests/test_operatorfree_additions.py tests/test_baseline_cli_chain.py \
  tests/test_standard_baselines.py tests/test_reference_operators.py \
  tests/test_graph_model_migration.py tests/test_runtime_contract_repairs.py \
  tests/test_fission_extension.py -q
```

完整回归结果：**371 passed**（43.17 秒）。活动 train/tune 的 `--help` 已确认三个入口，运行脚本 `bash -n` 通过；来源审计输出 18 个实现及固定来源哈希。

未运行正式训练或论文性能排名；数值/接口通过不代表精度已优于其他模型。

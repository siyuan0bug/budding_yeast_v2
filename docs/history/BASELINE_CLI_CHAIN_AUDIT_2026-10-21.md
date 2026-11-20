# 标准 baseline 上下游和 CLI 核查（2026-10-21）

## 当前选择与 U-Net

当前共 18 个入口：`mlp, resnet, unet, kan, transformer, mamba, deeponet, fno, wno, mwt, gnot, transolver, lno, neural_ode, sc_fno, bregman_fno, lamo, konode`。

U-Net 适合作为局部卷积和多尺度编码/解码对照。此前清理直接排除 U-Net 的范围过窄，本轮补回普通一维版本 `models/baselines/unet.py`。保留双卷积、MaxPool、转置卷积和 skip 拼接；只适配完整初值、ODE 参数、时间到多变量轨迹。不含 FiLM、图或额外注意力，不声称完全复现 RealPDEBench 中包含其他模块的 U-Net。

输入 `X:(B,V,2,T)`、`P:(B,D)`，输出 `(B,V,T)`。时间轴右侧补齐至 `2**n_blocks` 倍数并裁回。默认宽度 64、深度 4、39 变量、141 参数时有 10,849,383 个参数。HPO 为 U-Net 搜索宽度 16/32/64、深度 2–4、batch 32/64/128；实际 GPU 可用配置仍需按显存验证。

```bash
python train.py --model unet --dataset_name /absolute/path/to/dataset.npz \
  --width 32 --n_blocks 3 --loss_type hybrid_mse_rel --al_strategy none
MODEL=unet bash run_tune.sh
```

## 已核查及修复

| 环节 | 修复与证据 |
| --- | --- |
| train CLI → 构造器 | 使用实际 train.py parser 和嵌套生产 factory，逐个验证 18 个模型；DeepONet / NeuralODE 原来忽略的 n_blocks 现在控制隐藏层数 |
| 数据 → 训练 | 实际 DataModule，39 变量/141 参数的人工数据；每个模型实际 Lightning CPU 单步更新，检查参数变化 |
| HPO 构造器 | 逐模型执行 tune.py 实际构造调用，校验真实时间长度、深度、小波和 ODE 配置 |
| HPO → Bash → train | 修复导出多插一列 dataset_tag、未定义 DATASET_TAG 和浮点 lr 截断；统一为 16 列后附显式命名参数；实际 Bash read 后接实际 parser 校验 |
| 小波 / ODE / SC 参数 | 导出 n_blocks、weight_decay、dataset_tag，以及 WNO wavelet/level、ODE method/rtol/atol、SC 灵敏度文件和权重；run_tune.sh 传入对应配置 |
| pipeline 正常训练及 OOM 重试 | 两条启动命令均转发额外参数；附加参数摘要进入目录签名，避免不同深度/小波设置覆盖 |
| checkpoint → 推断 | 每个模型保存实际 Lightning checkpoint，用 eval.py 标准加载器严格加载；重载预测与保存前一致 |
| 下游分析/集成 | find_badcases.py 和 eval_ensemble_benchmark.py 改用显式 LABEL=PATH 和标准 checkpoint 加载器，按完整 preprocessing contract 重建数据；共同推断检查归一化和真实队列一致性；不再写死旧 resnet_film |
| 旧参数清理 | 移除活动 constructor_contract_version、tune 中不生效的旧 AL 参数，以及活动旧 baseline 注册/转发；训练/调参拒绝旧模型别名 |
| 历史结果 | 固定 CSV 的旧脚本放入 scripts/analysis/archive；旧模型指标和 checkpoint 不改名冒充当前模型 |

标准 baseline 不再搜索不参与其计算的 modes / wavelet level。ours 保留其自身入口与模块，本轮 baseline 清理不把 ours 里的合法名称误删。

## 验证结果与范围

```bash
python -m pytest tests/test_baseline_cli_chain.py tests/test_standard_baselines.py \
  tests/test_reference_operators.py tests/test_graph_model_migration.py \
  tests/test_runtime_contract_repairs.py tests/test_fission_extension.py -q
```

上一轮结果：**342 passed**。新增 KAN / Transformer / Mamba 后完整回归为 **371 passed**，配置见 [新增模型说明](OPERATORFREE_BASELINES_2026-10-21.md)。另在调整 U-Net HPO 显存搜索范围后重复对应采样测试：**1 passed**。覆盖非整除时间长度、BF16 前反向、输入依赖、来源核心数值对照、SC Jacobian、归一化/checkpoint、Fission 共享接口。CPU 警告主要为未启用 GPU、数据加载 worker 少和明确的历史 checkpoint 迁移提示。

train.py、tune.py、eval.py、两个下游工具分别运行 `--help`；两个运行脚本做 `bash -n`；来源清单执行 `--require-ready`。未直接启动用户的真实批量流水线，未做完整 GPU HPO 或整套论文性能实验。CLI 测试使用生产 parser/factory 而不是启动完整 train.py main，真实优化过程由 Lightning 执行。

SC-FNO 测试中的人工零导数仅用于检查数据传递；正式实验必须提供真实求解器参考灵敏度，不能复用这些人工值。

## 论文实验仍需独立处理的协议

当前 tune.py 沿用真实突变体物理 MAE 作为目标，包括相同队列的调参与模型选择。这不是独立测试协议。若论文把该队列称为最终测试集，须另设验证队列后再调参；本轮接口修复未改变既有实验划分，也不将其描述为无测试泄漏的 HPO。

18 个模型的默认参数量和 HPO 搜索空间不同。通过接口测试不等于已完成公平性能比较；正式实验需固定划分、输入信息、训练/调参预算与种子，并报告额外监督及计算成本。

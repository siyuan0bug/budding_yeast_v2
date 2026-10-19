# Train-only 归一化修复与未修改项解释（2026-10-19）

> **后续阶段**：用户随后已授权修复 pure-random RNG。下文“未修改 RNG”及故障探针记录
> 属于当时的历史阶段；当前修复见 [PURE_RANDOM_RNG_FIX_2026-10-19.md](PURE_RANDOM_RNG_FIX_2026-10-19.md)。
> corr 定义与重复样本分组仍未修改。

## 1. 本轮授权与实现

用户仅授权本阶段修复统计量拟合边界；等价输入分组切分、生成器 RNG 和 corr 定义先解释、不修改。
本阶段没有运行正式训练/HPO/real 性能重评，没有改写已有 NPZ、checkpoint 或历史指标。

`data/yeast_datamodule.py` 先保留原有 family/pattern 分层与 seed=42 的行切分，再进行：

- `fit_scope=train_synthetic`（新默认）：Y/P 均值、标准差仅用源 NPZ 的合成 train 行。
- `asinh_scale.s_c` 同样仅用 train；取固定 train 索引顺序前最多 20,000 条，逐变量跨样本/时间求
  90% 分位数，下限仍为 0.01。没有额外随机抽样或改变拆分 RNG。
- 之后冻结同一套变换，用于合成 val/test、real、物理损失、模型内部解码、AL 入库和独立评估。
- `log1p/linear/asinh_z/asinh_scale` 的变换公式、epsilon、恒定参数处理和逆变换保护均不改变。
- 默认仍是变量 `Z(log1p(max(Y,0)))`，参数 `Z(P)`，不是同时对 Y/P 取 log。

AL 的 train 指**源 NPZ 的合成训练拆分**，不是每一轮已经揭示标签的最小子集。
初始化子集、选样、real-envelope/nominal-flux 门控均未改变；后续 AL 继续复用初始拟合统计量。
因此修复的是 val/test 参与统计量拟合，不是完整的 AL 标签盲态或所有形式的数据泄漏。
数据指纹仍校验整个源文件，其哈希不参与数值拟合。

## 2. 新旧 checkpoint、实验和 HPO 隔离

`data/preprocessing_contract.py` 的版本 1 载荷已有显式 `fit_scope`，无需改变载荷版本即可区分：

- 新训练载荷标记 `train_synthetic`，恢复原统计量而不是重新拟合。
- 先前生成的 `all_synthetic_legacy` 载荷原样恢复，包括其旧范围和 `s_c`。
- 不允许显式指定与载荷冲突的拟合范围。也不允许把旧 all-synthetic 权重绑定新 train-only DataModule。
- 无完整预处理载荷的旧 checkpoint，`eval.py` 显式按 `all_synthetic_legacy` 重建，
  继承已记录的 DataModule split seed 和 norm_type，并继续警告其契约不完整。
  直接使用 Lightning/API 加载旧权重时，也必须给 DataModule 指定该历史范围。
  缺完整载荷时无法据此证明任意替换源文件与原训练一致，必须使用原数据。

新默认扁平运行名、层级超参签名和 W&B 分组加入 `trainfit1`，配置记录
`normalization_fit_scope`。显式 `--exp_dir` / `--name` 仍尊重用户路径，不自动迁移或清理旧结果。
新旧协议不能共用一个有历史 trial 的 HPO study：保留 real 物理 MAE 目标不变，
另记录并校验 `normalization_fit_scope=train_synthetic`，旧 study 需另起新名称/存储。

## 3. 为什么已经“去重”却仍重复

已量化核验的是指定的 `mutant_aware_pure_random_100000_210m_500s.npz`，不是所有数据集。
既有完整输入/轨迹哈希结果仍为 val 1,633/10,001、合成 test 1,604/10,001 与 train 重合，
real 3/126 重合；本阶段没有再次全量扫描或改变它们的切分。
证据：`scripts/analysis/outputs/pure_random_current_audit/input_duplicate_check/duplicate_inputs_summary.json`。

当前 pure-random 生成器确实调用 `is_duplicate_real_mutant`，但比较的是**候选 vs real 规则**，
不是**候选 vs 已生成合成样本**。任务字典以带递增编号的名称为键，不是最终 `(P, IC)` 的内容哈希。
离散 KO、拷贝数、碳源及重复组合均可能再次抽中；不同名称不能保证动力学输入唯一。
DataModule 再按行随机切分，没有按等价输入分组，所以等价样本可跨 train/val/test。

此外，当前字典去重存在源码已复现的别名/冗余字段漏洞：MAD2[CSL] 的 `Vi20` 和
`Vi20_active` 可表示同一有效动力学，却让字典比较不同。它说明字典去重并不等于最终输入去重，
但缺少原始生成代码版本时，不能把当前漏洞当作每条历史重复的完整生成证据。
其他生成器也有相似的 real 去重调用；这不证明其他 NPZ 的重合数量。
必须逐文件审计才知道，不能声称“仅这一份有问题”或“所有文件都已证实重复”。

## 4. 随机数问题不是必然写了两个不同的 seed

发生在 `generate_pure_random_dataset.py` 调用 `mutant_aware.py` 的链路：

- 主生成器创建 `rng=np.random.RandomState(seed)`；基因/模式等抽取用 `rng`。
- 连续幅度调用 `ma.sample_scaling_factor(mode,gene)`，没有传入 `rng`。
- 辅助函数内部用全局 `np.random.uniform/normal/beta/choice`；部分 mutation/rescue 分支也如此。

这是**两条独立随机状态**，并非必然显式写了两个数值不同的 seed。
只固定局部 RNG，并不能控制可能受先前调用影响的全局 RNG。已有 NPZ 不会因再次读取而变化。

本阶段人工探针（不积分、不训练）：局部 seed 固定 42，分别设全局 seed=10/11，按
`level=1+i%4` 生成 40 个候选；40 个描述相同，**32 个 override 字典不同**。
探针只用于复现，未修改生产 RNG；不同 level/调用序列的计数不可混为同一实验。

## 5. corr 的两个易误解点

`utils/metrics.py` 未修改。corr 主要回答“是否一起升降”，不回答“数值相差多少”。
例如真值 `[1,2,3]`、预测 `[2,4,6]`，corr 近 1，但每个时刻都存在浓度误差。

### 5.1 平直轨迹没有可比较的升降

| 真值 | 预测 | MAE | 当前 corr |
|---|---|---:|---:|
| `[5,5,5]` | `[5,5,5]` | 0 | 0 |
| `[5,5,5]` | `[10,10,10]` | 5 | 0 |

两行一条完全正确、一条明显错误，corr 却一样，因为目标无时间方差，Pearson 数学上未定义。
当前实现把零分子除以 epsilon，返回 0。这里低 corr 不能单独证明预测失败；须结合 MAE/MSE/轨迹。
真实计算还会对各变量 corr 求平均，恒定通道可拉低该平均值。

### 5.2 拼接后很高，不代表每条轨迹都很好

| 样本 | 真值随时间 | 预测随时间 | 单样本 corr |
|---|---|---|---:|
| A | `1→2→3`（上升） | `3→2→1`（下降） | −1 |
| B | `101→102→103`（上升） | `103→102→101`（下降） | −1 |

逐样本平均 corr = −1，两条轨迹的升降都反了；但直接把六个点拼接后，当前 pooled corr
约为 **0.999467**。原因是模型仍分清了“A 是低浓度、B 是高浓度”，这种样本间的大差异
掩盖了样本内部的反向变化。pooled 并不是先求逐样本 corr 再平均。
该例使用一个变量以隔离问题；实际代码先各变量拼接，再平均变量 corr。

## 6. 验证

新增回归覆盖四种模式：独立公式、train-only/显式 legacy 数值、分别只改变 val/test/real
后统计量及训练张量逐元素不变、改变 train 后统计量响应、legacy 载荷原样恢复、无载荷旧权重
禁止默配新统计量、fit_scope 校验、运行签名和 HPO 协议隔离。此前图/IC/物理/求解器回归一并运行。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_runtime_contract_repairs.py tests/test_relation_module_repair.py \
  tests/test_graph_model_migration.py tests/test_yeast_graph.py \
  tests/test_ic_wt_contracts.py tests/test_mechanistic_bifurcate_pino.py
```

最终结果：**385 passed，15 个预期的旧图/旧 checkpoint 迁移警告，31.25 秒**。
`train.py/eval.py/tune.py --help` 全部退出码为 0；保留 CRLF 的 whitespace diff 检查通过。
这些检查证明本次覆盖路径的工程一致性，不是 real-mutant 性能已经改善或所有数据集已无重复的证据。

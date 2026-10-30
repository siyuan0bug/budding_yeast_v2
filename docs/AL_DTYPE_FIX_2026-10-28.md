# AL跨轮候选输入精度修复（2026-10-28）

## 故障与影响

六项模型打分策略（Top-K、Power、MaxDist、LCMD、BAIT、Ours Response）在第一轮训练和评测成功后，第一次候选推理报`Input type (double) and bias type (float) should be the same`。Random不进行候选模型推理，原进程仍在正常运行，不应因该故障重启它。

根因在train.py逐轮重编码：将`_pool_initial_physical`（float64来源初值）直接传入保持输入dtype的normalize_y，得到float64 `_pool_Y_ic`；与时间轴stack后输入仍是double，FNO权重为float32。普通静态训练的DataModule已正确使用float32，因此第一轮训练有效。之前测试包含精确初值保存和多轮训练，但遗漏了真实float64来源初值在逐轮重编码后进入候选推理的组合场景。

## 修复

- `pool.refresh_model_inputs`统一负责首次加载与每轮重编码；仅在本地模型编码输入上转float32，物理初值、参数来源及文件不改写。
- train.py每轮调用统一接口，清理旧模型输入和特征缓存；参数逐维Z-score、变量log1p+Z-score、train-only拟合规则不变。
- 时间轴显式匹配模型IC的dtype，避免stack隐式提升精度。
- 百万池的mutation及有效输入校验增加每10万条进度日志。仍完整校验，不跳过real碰撞规则，也不重新生成候选。

## 验证与边界

101项相关测试通过，覆盖多轮重新拟合、池裁剪、全部7个当前策略、Fission接口和标签；后续输入缓存失效/批调度检查38项通过（集合有重叠，不合并计数）。

持久工具`scripts/analysis/audit_al_pool_inference.py`读取本次LCMD第一轮真实FNO checkpoint、Uncertainty第一轮独立成员和真实pool。先用旧表达式成功复现double错误，再验证修复后：128个候选、16个合成训练锚点、每策略4个选择，7种策略全部成功；输入float32，来源初值仍为float64，稳定候选ID可正确解析。仿真入口被断言禁止，目标轨迹置NaN防止提前读取。只做预测/选择，不重新训练，不执行ODE，不代表百万池整轮吞吐或显存峰值已测。

报告：`/home/users/zsy/mydata/budding_yeast/analysis/al_dtype_fix_2026-10-28/checkpoint_acquisition_audit_resolved.json`。原第一轮checkpoint、评测、数据快照及Random进程均保留。

## 六项失败实验的重新启动

当前没有完整AL断点续跑（尤其候选池消耗与RNG状态恢复），本次不将部分输出冒充精确续跑。六项失败实验可从相同初始数据重新训练到新的结果根目录；第一轮旧结果保留，不覆盖。此命令已做DRY_RUN检查，正式启动交给用户：

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
SKIP_PREPARE=1 MAX_JOBS_PER_GPU=2 \
EXP_ROOT=/home/users/zsy/mydata/budding_yeast/experiments/al_dtype_retry_20261028 \
STRATEGIES="al4pde_uncertainty al4pde_power al4pde_max_dist al4pde_lcmd al4pde_bait ours_response" \
bash run_al_budding.sh
```

新结果写到`.../experiments/al_dtype_retry_20261028/al_budding/`，共6项，GPU0/1/2/3任务数1/2/2/1，与原Random分开。原池和初始数据不需要重新生成；W&B继续强制offline。本修复没有修改ODE、mutation目录/概率、过滤、配额或训练预算。

## 2026-10-29 当前新候选池复验

使用当前gene_ranges百万pool与新共享初始集，再次用旧LCMD第一轮真实FNO checkpoint及其独立ensemble成员，在CPU上运行`audit_al_pool_inference.py`：128个真实候选、16个新训练锚点，每策略选择4个，Random/Uncertainty/Power/MaxDist/LCMD/BAIT/Ours Response全部通过。旧逐轮表达式仍能复现double/float报错；统一刷新后模型输入/参数均float32，来源初值仍float64。15项专门dtype回归重跑通过，涵盖Budding/Fission形状、4种归一化、重复重拟合及池裁剪。

这是小池推理/选择验证，不是百万池全轮吞吐或新模型精度验证；本次随机128候选的Ours Response没有同背景对照边，因此该项走探索回退，非空对照分支由已有专门测试覆盖。没有训练、没有ODE查询或云端W&B调用。报告为`/home/users/zsy/mydata/budding_yeast/analysis/gene_ranges_preparation_2026-10-29/current_pool_dtype_audit.json`。

用户贴出的`refit_balanced/.../round_00`报错是保留的旧实验记录；当前launcher默认新`refit_gene_ranges`目录。正式训练须启动新进程加载修复，不会自动改写旧错误日志。

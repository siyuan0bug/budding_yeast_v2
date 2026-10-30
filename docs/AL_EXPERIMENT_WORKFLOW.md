> **2026-10-30 无损存储更新**：初始数据共享、追加数据只存一次、各轮保存`dataset.json`清单。完整样本内容、顺序、切分和归一化不变；详见[新布局与迁移验证](AL_DATASET_STORAGE_LAYOUT.md)。

> **2026-10-30 正式长程批次已启动**：默认切换为research全部14项FNO＋full档（100/100/100/200 epochs、warmup5/5/5/10），GPU0–3队列4/4/3/3，每卡最多2项。保留screen作为显式选项。tmux后台与结果位置见[本批启动记录](AL_FULL_LAUNCH_2026-10-30.md)。此条表示已启动，不表示实验已完成。

> **2026-10-30 最新预算档位**：默认screen采用50/50/50/100 epochs和warmup4/4/4/8枚举方法；显式TRAINING_PROFILE=full采用100/100/100/200和5/5/5/10复核少数候选。两档均保留峰值LR1e-3、最低LR1e-5、FNO192/4/24、batch512和loss变化率监控，不自动早停，不保证充分收敛。尚未正式训练。[设置、目录与验证](AL_TRAINING_BUDGET_2026-10-30.md)。下方较早配置与结果保留为历史记录。

> 当前Budding后续AL方法/模型比较固定使用gene_ranges百万pool和seed42共享initial。输入文件SHA256及采样源码身份已记录在 `/home/users/zsy/mydata/budding_yeast/analysis/budding_al_fixed_inputs_2026-10-29/fixed_input_manifest.json`。同一初始/切分seed复用相同输入；各方法后续查询独立。采样域固定不代表覆盖所有real邻域，L4仍域外；TEM1越界仍需作为独立数值质量问题处理，不在运行中的对照实验中悄悄改变仿真规则。

> 2026-10-29 15:33：Random、Uncertainty、ours_response已按修复后的代码后台重跑，旧失败记录整体保留。MaxDist/LCMD/BAIT/Power已完成，用时约3:36/3:39/4:20/4:22。[本次重跑与用时记录](AL_GENE_RANGES_RESTART_2026-10-29.md)。

> 2026-10-29 后续修复：ours整数序列化及源头池大小已修复；Random/Uncertainty阻塞已取得满唤醒管道证据，并独立复现Python3.10锁循环。有界提交通过验证，train.log新增逐行北京时间和阶段耗时。[日志与阻塞修复说明](AL_LOGGING_AND_PROCESS_POOL_FIX_2026-10-29.md)。旧进程不会热更新。

> **2026-10-28 候选推理dtype修复**：逐轮归一化曾将float64来源初值带进float32 FNO，导致六项策略在首轮选样失败。现在首次加载/逐轮重编码统一调用`refresh_model_inputs`，物理初值保留float64、模型输入float32；真实checkpoint与7策略小池推理已验证。Random原任务保留；六项失败任务的新目录重启命令见[故障修复](AL_DTYPE_FIX_2026-10-28.md)。未自动重启正式训练。

> **2026-10-28 同卡并行**：Budding默认`MAX_JOBS_PER_GPU=2`，每卡最多两项完整实验；当前七项分布2/2/2/1，可最多七项并发。设为1恢复同卡串行。按进程完成事件补位，退出码按完成顺序记录，失败默认继续；`CONTINUE_ON_ERROR=0`停止启动该卡后续任务、等待已运行任务结束。显式中断清理全部受管子进程。需要Bash≥5.1（当前服务器5.1.16）。`WAIT_FOR_GPU=1`只在该卡没有本批活跃任务时等空闲，第二个自有槽位不被idle阈值阻塞。并发数为固定上限，不根据实时剩余显存无限扩张；Fission脚本不变。
>
> 结果仍在`/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno/<loss_strategy>/w192_n4_m24_b512_e50-100_lr1e-3_wm4-8_al4_refit_balanced/seed_42/`，通过`refit_balanced`配置子目录区别历史结果；本次未启动正式训练。并发任务独立读取同一初始集/池，追加样本、checkpoint、W&B offline均写各自目录。ODE默认每实验`AL_SIM_WORKERS=4`，CPU线程默认OMP2、BLAS1、MKL2（显式环境变量保留）。显存峰值含候选预测和选择，不能只依据训练阶段占用；同卡共享计算也会影响墙钟时间，不作为独占性能基准。

> **2026-10-28 Ours接入**：默认FNO六个基线加`ours_response`，四卡队列2/2/2/1。共用balanced百万池和初始10k/1250/1250固定划分；本次只准备数据，未启动训练。Ours数学与审计字段见[实现说明](FNO_AL_BASELINES_AND_OURS_2026-10-28.md#11-ours_response-首版实现)。

# Budding / Fission 主动学习实验

> **统一参考配置（2026-10-30按耗时要求更新）**：Budding默认screen=50/50/50/100 epoch、warmup4/4/4/8；full=100/100/100/200、warmup5/5/5/10。两档batch512、峰值LR1e-3、最低LR1e-5。所有方法和辅助ensemble在同档位比较。Fission不变，未启动新训练；短档仅为有限预算比较，不能宣称充分收敛。

> **2026-10-28 FNO/AL研究配置**：当前`run_al_budding.sh`仅运行FNO的6个固定池基线（Pool Random、Top-K、Power、预测轨迹MaxDist、LCMD、BAIT）；加上`ours_response`共7项，四卡队列2/2/2/1。新增MaxDist使用实际官方bmdal选择器，旧自定义diversity不作官方基线。Ours首版条件响应不确定性已实现，使用独立ensemble及合成训练锚点，收益尚未验证，详见 [FNO基线与研究设计](FNO_AL_BASELINES_AND_OURS_2026-10-28.md)。本轮未启动正式训练。

> **当前批实验配置**：直接运行现有`run_al_budding.sh`，默认复用已生成的balanced百万池；balanced初始集已生成并在当前7项FNO实验间共享。结果目录的设置段使用`_refit_balanced`，W&B默认开启且强制offline。仅更新原脚本配置，本轮未启动数据准备或训练。

> **新池badcase覆盖实测**：百万池已生成；固定Top20严格局部25%邻域计数合计155→38，#15为1→2、#22为0→1，多个L2背景反而减少。尚未训练新模型。完整20例与新旧变化的混杂因素见 [覆盖对比](BALANCED_AL_POOL_COVERAGE_2026-10-28.md)。

> **2026-10-28 新配额已接入**：Budding AL采用模块集合等额、集合内合法模式容量感知均衡；新百万池为9815/126225/863960条L1/L2/L3，默认文件改为`budding_pool_1000000_seed42_balanced.npz`，初始集使用独立`balanced`路径。静态与Fission不变，本轮未启动训练。实现、有限容量与分布说明见 [两级均衡候选池](BALANCED_AL_POOL_2026-10-28.md)。

> **2026-10-28 样本内容补齐**：静态与AL统一保存显式mutation列表、overrides、精确初值和来源ID；AL累计快照/新增文件不再丢失这些元数据。生物标签、ABC及中性六分类均保留，未仿真池不伪造标签，Fission标签不变。字段、适用范围及历史文件说明见 [样本内容契约](DATASET_SAMPLE_CONTRACT.md)。

> **2026-10-28 后续设计状态**：Budding AL非GAL环境已按用户确认改80/15/5。模块集合等额配额已接入并生成balanced百万池；用户确认的batch512、50/50/50/100 epoch、warmup4/4/4/8已写入Budding批脚本所有默认模型/策略行，Fission不变；详见 [配额与官方训练预算](AL_POOL_REDESIGN_AND_TRAINING_BUDGET_2026-10-28.md)。旧池和原实验未改写。

> **2026-10-28 已有实验完成度与用时**：当前Budding批次已启动8项、完成4项，其余4项中断；实际训练epoch100/100/100/200。当前候选池初选、过滤后分布和旧目录版本的区别，以及各完整实验墙钟用时见 [候选池与用时核对](AL_POOL_AND_RUNTIME_2026-10-28.md)。本轮未恢复训练或生成新的正式候选池。

当前入口为独立的 `run_al_budding.sh` 和 `run_al_fission.sh`，参数、数据准备和调度直接放在各自脚本中；闭环训练在 `train.py`。Budding 无额外 baseline 启动器或共享 shell 转发层。旧生成器和已有数据不改写。

实验结果默认统一存到机械盘 `/home/users/zsy/mydata/budding_yeast/experiments/`，其下 `al_budding/` 和 `al_fission/` 分别保存两系统的 AL 大实验。两个 AL 脚本、`run_pipeline.sh`、`train.py --exp_root` 与汇总工具共用该默认根目录；`EXP_ROOT` 环境变量或显式 `--exp_dir` 可覆盖。直接训练的旧式扁平结果默认位于该根目录的 `flat_runs/`。`DATA_DIR` 只控制 pool/初始输入，不改变结果根目录。

2026-10-23 按用户要求删除数据根目录中两份旧 AL 输入 `pure_random_1m_pool.npz` 和 `al_iter0_10000_210m_500s.npz`（共654,843,196字节）；路径、大小与SHA256记录在机械盘 `al_storage_cleanup_2026-10-23.json`。历史实验内的快照保留。默认输入文件名已去掉 `_v2` 后缀，使用 `_al.npz`；规则版本保留在 NPZ 内部元数据，复用时仍做校验。

```bash
cd budding_yeast_v2
GPU_ID=3 MODEL=fno STRATEGIES=al4pde_random bash run_al_budding.sh
GPU_ID=3 STRATEGIES=al4pde_random bash run_al_fission.sh
```

两个入口采用与 `run_pipeline.sh` 相同的配置形式：`SEEDS=(42)` 和逐行的 `EXPERIMENTS=(...)`。每行指定一个完整 AL 实验及 GPU，每个 seed 执行清单中的全部实验；Budding同卡默认最多2项并行、跨卡并行，Fission仍同卡串行。Budding默认7行（FNO×6个AL基线+ours_response），Fission默认4行（标准fno×四策略），均使用GPU0–3。Budding基线为Pool Random、Ensemble Top-K、Power、预测轨迹MaxDist、LCMD、BAIT；Fission默认四策略保持Random、Uncertainty、Power、BAIT。可直接增删、注释行或修改任意行的 GPU 和超参数。`MODEL/MODELS/STRATEGIES` 仅过滤清单已有行，`GPU_ID` 可统一改为单卡。两个 AL 脚本强制 `WANDB_MODE=offline`，训练 logger 对 AL 同时设置 offline，禁止上传云端；即使外层传入 `WANDB=1` 或 `WANDB_MODE=online`，AL 仍离线。W&B 在各实验目录的 `wandb/offline-run-*/` 保存二进制指标、配置与媒体；不要执行 `wandb sync`。`PYTHON_BIN`、`EXP_ROOT`、`DATA_DIR`、`POOL_FILE`、`INIT_DATASET`、`SIM_WORKERS`、`AL_SIM_WORKERS` 等可通过环境变量覆盖。已有实验目录拒绝混写，需换 `EXP_ROOT` 或 seed；本次未新增中断恢复功能。

| 设置 | Budding | Fission |
|---|---:|---:|
| 默认候选池大小 | 1,000,000 | 500,000 |
| 候选池 seed | 42 | 20000 |
| 默认选样/切分/训练 seed | 42 | 42 |
| 初始合成总量 | 12,500 | 6,250 |
| 初始 train / val / test | 10,000 / 1,250 / 1,250 | 5,000 / 625 / 625 |
| 独立 real 评测 | 126 | 20 |
| 四轮实际训练目标 | 10k / 20k / 40k / 70k | 5k / 10k / 20k / 35k |
| 三次新增有效轨迹 | 10k / 20k / 30k | 5k / 10k / 15k |
| 四轮 epoch | 所有默认模型: 50/50/50/100 | FNO: 60/60/60/120 |
| warmup | 所有默认模型: 4/4/4/8 | FNO: 6/6/6/12 |
| 起始 / 最低学习率 | 1e-3 / 1e-6 | 1e-3 / 1e-6 |
| 启动器默认模型 | fno | fno |

Fission 池逐候选执行原生预热，不是零仿真成本；主轨迹仍在选中后求解，保持原生 BIO/Pattern（包括缺失 Pattern）和初值规则。

## Budding FNO与六种AL基线

入口`run_al_budding.sh`：GPU0并行Pool Random和MaxDist，GPU1并行Ensemble Top-K和LCMD，GPU2并行Power和Ours Response，GPU3运行BAIT，共7个独立四轮FNO实验。先统一验证/准备一个候选池和每seed一个初始集，再启动四条队列。所有策略共享相同seed的初始数据和固定留出集。

| 模型 | width | n_blocks | batch_size | LR | modes |
|---|---:|---:|---:|---:|---:|
| fno | 192 | 4 | 512 | 1e-3 | 24 |

当前固定模型和损失比较AL选样方法，不把不同网络容量混入主对照。直接修改EXPERIMENTS行，或使用WIDTH/BATCH_SIZE/N_BLOCKS/LR/MODES覆盖筛选后的任务；batch变化后不自动重算epoch/warmup。完整候选基线映射、未实现的latent coreset以及ours方案见[FNO研究配置](FNO_AL_BASELINES_AND_OURS_2026-10-28.md)。

Budding统一batch512、50/50/50/100 epoch、warmup4/4/4/8；每轮重建余弦调度，最低LR1e-6、weight decay1e-4不变。Top-K/Power所需辅助ensemble训练成本单独计入，最终评测仍使用主模型。Fission当前配置未改。没有在本轮启动正式训练。

同类 FNO 以标准 baseline 的总隐藏通道为参考，modes=24、主干 blocks=4；Budding 起点 width=192，Fission width=128。`cross_fno` 已退出默认实验清单，不再按其每变量通道数推导其他模型的宽度；本次未重写模型 API 或历史 checkpoint。

### RTX A6000 与 biom 参考

2026-10-23 实机只读查询：每卡为 NVIDIA RTX A6000，`memory.total=49140 MiB`（约 48 GiB），官方标称 48 GB；GB 指字节，Gb 指比特，48 Gb 只有约 6 GB。[NVIDIA 规格](https://www.nvidia.com/en-gb/products/workstations/quadro/rtx-a6000/)。正式实验每张卡同一时间只运行本批次的一项任务，其他进程的占用仍会影响可用显存。

以下为 CPU 仅构造计数，复参数按两个实数计；没有进行 GPU 前向或训练：

| 系统/模型 | width | 主干 n_blocks | 实标量参数量 |
|---|---:|---:|---:|
| Budding fno | 192 | 4 | 7,290,791 |
| Budding deeponet | 256 | 3 | 2,941,735 |
| Budding unet | 64 | 4 | 10,849,383 |
| Budding transformer | 192 | 4 | 1,375,143 |
| Fission fno | 128 | 4 | 3,240,080 |
| Budding biom_refiner_plus（可选） | 192 | 4 个细化块 | 12,860,055 |
| Fission biom_refiner_plus（可选） | 128 | 4 个细化块 | 5,519,881 |

biom_refiner_plus 的 width 是每条主干流的隐藏通道，不乘变量数量；它有双流和两阶段，默认 4 个粗预测块另加 CLI n_blocks 控制的细化块。历史记录中大量已评测配置采用 width192/modes24/batch256/LR8.8e-4，也有 batch512 的历史配置，但不构成当前 AL 配置的精度或显存保证。两个脚本内保留了可取消注释的 biom_refiner_plus 参考行，LR8.8e-4；默认任务数为 16/4。

参数量不代表显存峰值：上述模型的参数、梯度和 Adam 双动量占用只是百 MiB 量级，512 条×500 步的中间激活、反向缓存和临时工作区才是主要开销。当前 `bf16-mixed` 不意味着所有张量均为半精度，FNO 的 FFT 明确使用 FP32；Transformer 的 `need_weights=False` 允许优化 attention 路径，但本次未测最终后端。百万池 LCMD/BAIT 的特征和选择内存还要单独计入。以上属于适合 48GB 级单卡的起始范围，未实测正式训练峰值，不能宣称必定无 OOM。

### 2026-10-23：用户授权的显存实测

在空闲物理 GPU 5 上逐模型独立进程执行：batch512、500 时间步、BF16 mixed、hybrid_mse_rel、真实 `training_step`（BioRefiner 包含粗预测辅助损失）、反向传播、梯度裁剪和 AdamW 更新，共两步以覆盖优化器状态初始化后的峰值。输入为固定形状的合成归一化张量，不运行完整 AL、不上传 W&B。

| 系统 | 模型 | width | 张量峰值 GiB | 分配器预留峰值 GiB |
|---|---|---:|---:|---:|
| Budding | fno | 192 | 3.19 | 4.10 |
| Budding | deeponet | 256 | 1.22 | 1.39 |
| Budding | unet | 32 | 0.97 | 1.09 |
| Budding | transformer | 192 | 8.81 | 10.69 |
| Budding | biom_refiner_plus | 192 | 32.33 | 34.65 |
| Budding | biom_refiner | 192 | 25.76 | 29.12 |
| Budding | hyper_fno | 192 | 23.23 | 24.59 |
| Fission | fno | 128 | 2.07 | 2.62 |
| Fission | biom_refiner_plus | 128 | 16.95 | 17.96 |
| Fission | graph_resnet_film | 128 | 8.27 | 8.78 |

十项均完成两步，无 OOM。预留峰值包含 PyTorch allocator 缓存，不包含 CUDA context、其他库或其他进程；完整训练的验证、ensemble 和候选选择仍需单独计入。GiB=2^30字节。Budding biom_refiner_plus 已占约35 GiB预留，不建议直接把 batch 翻到1024；本次没有改变默认实验配置。

复现：`python scripts/analysis/profile_gpu_memory.py --gpu 5 --batch 512 --steps 2 --output <新的输出目录>`。逐项 JSON、日志与汇总见 `scripts/analysis/outputs/gpu_memory_2026-10-23/`。本次是显存测量，不是精度或收敛实验。

补测 Budding batch1024，其他条件相同：FNO192 预留峰值 8.10 GiB、DeepONet256 2.60 GiB、UNet32 2.12 GiB、UNet64 3.97 GiB，均成功完成两步。结果位于 `scripts/analysis/outputs/gpu_memory_2026-10-23_larger_batch/`。可用 `--system budding --model unet --width 64 --batch 1024` 选择配置；轻量模型默认已据此调整为1024，Transformer及较重ours仍512。Fission FNO128的batch1024本次未单独GPU补测，不将Budding补测当作其实际峰值。

HyperFNO 当前显式构建逐样本复数谱权重 R_dyn，形状 `(B,width,width,modes)`；B512/W192/M24 的每层单份 R_dyn 为 3.375 GiB complex64，四块加反向会显著增加显存。BioRefiner 的粗阶段双流/多分支、细阶段、未截断的粗阶段计算图，以及变量注意力中间张量共同增加激活开销。UNet 的32表示基础通道（32/64/128/256/512），而不是所有层只有32通道；不同模型的width不能按打印数值衡量规模。

大batch可用于学术实验，但需报告并调整优化预算：初始10000样本在batch512/1024/2048时分别约20/10/5步每epoch；不能在相同epoch下将三者视为相同更新次数。同一模型比较不同AL策略时固定batch/LR/训练计划，跨模型可采用各自调参后的batch，并透明报告。[Shallue等大batch研究](https://arxiv.org/abs/1811.03600)。

每行固定 13 列：

```bash
# MODEL LOSS_TYPE AL_STRATEGY MODES WIDTH N_BLOCKS BATCH_SIZE LR ROUND_EPOCHS FINAL_EPOCHS ROUND_WARMUP FINAL_WARMUP GPU_ID
SEEDS=(42 1024)
EXPERIMENTS=(
    "fno hybrid_mse_rel al4pde_random      24 192 4 1024 1e-3 100 200 8 16 0"
    "fno hybrid_mse_rel al4pde_uncertainty 24 192 4 1024 1e-3 100 200 8 16 1"
    "unet hybrid_mse_rel al4pde_random     24 64  4 1024 1e-3 100 200 8 16 0"
)
```

上述示例为 3 行×2 seed=6 个四轮实验，GPU 0 有四项、GPU 1 有两项，各卡按MAX_JOBS_PER_GPU上限执行（默认2，设为1则串行）。Fission FNO 按相同列格式配置，默认 epoch 为60/120、warmup为6/12。也可用 `SEED_LIST="42 1024"` 覆盖种子数组。

```bash
# 仅打印 20 项计划：不生成数据、不训练
DRY_RUN=1 bash run_al_budding.sh
# 用户正式启动，需事先登录 W&B；Budding 默认 WANDB=1
GPU_IDS="0 1 2 3 4" SEED=42 WANDB=1 SIM_WORKERS=8 AL_SIM_WORKERS=4 bash run_al_budding.sh
```

可用 `MODELS="fno transformer"` 限定清单已有模型；`MODEL=fno GPU_ID=0` 将选出的行放到 GPU 0 顺序运行。单独设置 `MODEL` 不修改清单内的 GPU 分配。`GPU_IDS` 是按筛选后的策略顺序统一映射 GPU 的快捷方式；直接编辑每行的 GPU 字段不受此限制。外层队列日志保存在 `<EXP_ROOT>/al_<system>/batch_logs/<批次>/gpu<N>.log`；`tasks.tsv` 汇总每项实验的 seed、模型、策略、退出码和结果路径。每个独立实验仍有自己的 `train.log`。单项失败默认记录错误、继续该卡下一项，批次最终返回非零；`CONTINUE_ON_ERROR=0` 可在失败后停止该卡队列。不会自动修改 batch size 重试或覆盖旧 checkpoint。

候选池每批准备一次；每个 seed 分别准备并复用自己的初始数据文件。显式指定 `INIT_DATASET` 且运行多个 seed 时，路径必须带 `{seed}` 占位符，避免错误共用切分。`PREPARE_ONLY=1` 仅准备共享数据，`DRY_RUN=1` 只显示计划。重复结果路径会在数据生成前拒绝。Ctrl+C/SIGTERM 会终止本次启动器跟踪的 worker 和子进程树；此退出处理不等于 AL 断点恢复。

W&B 每个模型×策略×seed 使用一个 run，四轮训练指标采用 `round_00/...` 至 `round_03/...`，横轴分别为对应的 epoch；`al/real_mutants_mae`、RelL2 等指标以 `al/train_samples` 为横轴。已修复 Lightning 默认 `-` 前缀分隔符与原 `/` 横轴定义不匹配的问题；此前本地 JSON 不受此命名问题影响。辅助 ensemble 成员保存在本地，W&B 主曲线记录主模型，不代表辅助成员均值。

## 初始数据来源与切分

新入口 `python -m budding_yeast_v2.data.systems.al_initial --pool <pool.npz> --output <initial.npz> --train-samples 10000 --seed 42 --workers 8`。

1. 从当前参数池的 L1/L2/L3 分别选点，总配额为 5%/40%/55%。每层运行 MiniBatchKMeans，使用整个候选池的参数均值/标准差，按逐维 Z-score `(P-mean)/std` 参数距离挑选每簇最近的实际候选；不是直接仿真聚类中心。接近常量的列使用分母 1，避免除零。生成报告记录 `initial_parameter_transform=zscore` 和 `initial_parameter_fit_scope=full_source_pool`。此前生成的 asinh 选点数据不会自动重选；新规则需要生成新的初始文件。

启动器复用已存在的初始文件前，会检查上述选点规则、候选池路径、训练样本数和 split seed；不匹配时退出并要求选择新的 `INIT_DATASET` 路径，不覆盖已有文件。该检查不替代候选池内容哈希校验，请勿原地替换候选池文件。
2. 对全部初始候选仿真，失败在原 L 层补采，有界尝试不足则报错。成功轨迹按候选 ID 排序，消除 worker 完成顺序影响。
3. 对有效合成轨迹用局部 `RandomState(seed)` 随机排列，按 80%/10%/10% 切分，完全不依赖 BIO、Pattern 或 L 标签。12500/6250 能得到表中的准确数量；L 精确配额只约束合成总集，各拆分可有随机波动。
4. 添加完整的原生 real 基准，单独保存 real 索引，不参与随机切分。文件保存 `split_train/val/test/real`、`split_seed`、`split_policy`。
5. 初始 train/val/test 及已尝试失败的候选均从后续池移除。后续新增只进入 train，固定 val/test/real。

默认文件位于 `/home/users/zsy/mydata/budding_yeast/` 或 `/home/users/zsy/mydata/fission_yeast/`：`<system>_pool_<N>_seed<pool_seed>_al_v2.npz`、`<system>_initial_train<Ntrain>_seed<seed>_al_v2.npz`。候选池没有完整轨迹。初始文件包含 train/val/test 和 real。原 `data/al_pool_manager.py` 不再是新启动器的初始集入口。

固定池及相同的初始化/切分 seed、配额和版本规则时，所有模型和 AL 方法复用完全相同的初始文件及 train/val/test 索引；单独固定 pool 不能保证不同 seed 的初始数据相同。每个池只需生成一次，每个初始化 seed 的初始轨迹文件只需生成一次。`PREPARE_ONLY=1 bash run_al_budding.sh` 或对应 Fission 命令可预先准备这些文件，后续启动只校验复用。各实验引用同一不可写初始块，并保存每轮数据清单用于复现；主动学习追加数据按每个实验自己的选择单独生成并保存一次，不再复制初始文件或累计轨迹。

共享数据回归已核对同一 seed 下 20 个任务引用相同 pool 和初始文件，且初始生成只调用一次。未启动正式训练或新增 GPU 短跑；模型 API 的历史 width 语义未改变。

## 每轮训练、归一化与补采

每轮冷启动模型、优化器、warmup 和余弦退火；模型内部 curriculum 使用当轮 epoch。Uncertainty / Power 的独立 ensemble 成员使用当前轮同一数据、尺度和 epoch，最后一轮不训练选样辅助成员。

每轮从累计物理训练轨迹和参数重新拟合归一化，禁止用 val/test/real 拟合。以当轮统计量重新编码固定留出集、real 和候选池。每轮`dataset.json`清单准确还原全部累计训练数据、固定留出数据和拆分索引；checkpoint 和 `preprocessing.pt` 绑定其还原的数据及该轮统计量，单独评估使用对应的 `rounds/round_XX/dataset.json`，旧NPZ仍兼容。不会对已经截零的归一化轨迹做逆变换来重建原始样本。

新增预算为初始训练量乘以追加序号 1、2、3。数值失败/超时后的候选被消耗移除，按原策略继续选样补足有效轨迹，池不足则明确报错，不静默欠额。补采不强制 L 配额；L 比例约束的是池和初始采样。各策略选样打分及 ours 参数门控未更改。

归一化每轮变化，因此 normalized loss 在跨轮大图中分面显示，不解释为同一物理尺度的连续损失；物理空间误差可以跨轮比较。W&B 同一次运行中使用 `round_XX` 指标前缀和对应 epoch 横轴，保留原生标签统计以及 L 统计。

## 输出

```text
/home/users/zsy/mydata/budding_yeast/experiments/
  al_budding 或 al_fission/<model>/<loss_strategy>/<hyperparameters>/seed_42/
├── config.json / graph_provenance.json / train.log / wandb/
├── data/
│   └── pool_manifest.json  # 引用共享初始块；不复制轨迹
├── rounds/round_00 ... round_03/
│   ├── dataset.json  # 引用初始块及此前全部新增块
│   ├── preprocessing.pt / labels.json
│   ├── model.ckpt
│   ├── training_history.csv
│   ├── loss_curves.png / loss_curves.pdf
│   ├── ensemble/ 与 ensemble.json（仅需要 ensemble 的追加轮）
│   └── eval/
│       ├── metrics_denorm.json
│       └── real_mutants/（每轮完整 real 预测图）
├── acquisitions/after_round_00 ... after_round_02/
│   ├── selected.csv
│   ├── added_samples.npz
│   └── report.json
└── summary/
    ├── al_performance_curve.csv / al_performance_curve.png
    ├── all_rounds_loss.png / all_rounds_loss.pdf
    ├── rounds_summary.json
    └── global_metrics_denorm.json
```

每轮 `eval/` 只有 real 图片和指标 JSON；JSON 包含 real/合成测试/整体指标，以及逐样本物理指标。最终 JSON 与最后一轮 JSON 相同。保留四轮原有风格 loss 图，不覆盖；跨轮大图按 round 顺序复用同一绘图实现。训练 loss 在训练 epoch 结束后读取最终归约值，避免首个 epoch 缺失和一轮滞后。

所有新累计快照和新增数据都保存逐行 `levels`、`requested_levels`、`module_counts`。合成样本满足三者相等且属于 1–3；real 使用 level/requested=0、module_counts=-1，不代表其实际突变阶数为零。相同基因复合模式算一个模块，MDT/Growth 不算基因模块。累计快照现在包含固定验证/测试集，应按存储索引读取，不能再随机切一次。

## 验证范围

357 项回归通过。两种系统分别实际生成带完整 126/20 real 基准的初始小数据，验证当前按层 KMeans、5/40/55 配额、seed 随机切分和有效轨迹。随后各用 8 条合成训练样本、2 株 real 子集和小型 MLP 跑完四轮（epoch=1/1/1/2），实际训练量为 8/16/32/56，三次新增 8/16/24。八个 checkpoint 均独立重载，固定留出轨迹逐元素不变，各轮统计量变化且与当轮累计 train 一致；L 字段、每轮 real 图/JSON、单轮/跨轮 loss 文件均检查通过。`eval.py --exp_dir` 自动找到最后一轮模型和数据后，物理指标与保存结果完全一致；汇总工具已接入新 summary 路径。

这些是流程验证，不是正式精度实验，也未启动默认规模的长时间训练。验证记录在 `scratch/probes/al_experiment_workflow_2026-10-22/`。

独立复评可运行 `python eval.py --exp_dir <seed目录> --device cuda`，自动取最新可用 `round_XX/model.ckpt` 和同目录 `dataset.json`（或历史`dataset.npz`）；指定其他 round 时可直接给 `--ckpt <round目录>/model.ckpt`。旧 checkpoint 没有相邻数据文件时仍须显式指定 `--dataset_name`。

## 正式实验前的剩余检查

2026-10-22：初始化 Z-score 与旧初始文件复用检查已覆盖，流程、ensemble、Power/BAIT、LCMD 相关 51 项回归通过；这不代表已完成默认规模的性能验证。

- 用户决定不安排额外短跑，正式实验由用户启动。默认规模采集成本尚未实测；五个默认策略实际扫描完整池，不能把 `al_pool_subset_size=10000` 理解为其计算量上限。LCMD/BAIT 选择阶段成本仍需在正式运行时观察。
- 同一系统、同一 seed 的所有策略复用相同候选池、初始训练集和固定验证/测试集。仅初始集按 L 比例选点，后续由策略决定 L 分布，不额外强制配额。
- Uncertainty/Power 默认训练两个独立成员，评测主模型；记录额外训练和采集耗时，区分有效仿真预算与计算预算。正式比较建议多个 seed。
- 每轮重新拟合归一化，归一化 loss 跨轮不可直接作为统一误差尺度；横向比较以反归一化指标为主，同时保留逐 real 指标以检查 badcase。
- 当前配置记录 Git commit，但不能完整还原尚未提交的修改；正式实验前保存代码版本，并固定数据文件、记录哈希。不要原地替换池文件。
- 当前没有完整的 AL 中断恢复；中断后不能在旧目录上直接续跑。实验盘空间、GPU 和仿真 worker 数应在运行前确认；静态/初始生成使用 `SIM_WORKERS`，在线追加使用 `AL_SIM_WORKERS`。

2026-10-22 批量入口和 W&B 修改：121 项自动化测试通过，覆盖模型接口、CLI 链路、四轮数据约束、20 项队列配置与前缀/横轴契约。队列测试使用替身命令验证调度，不启动正式实验；W&B 在线服务、登录和网络未验证。

随后将 Budding 全部 shell 配置与调度合并至 `run_al_budding.sh`，移除额外 baseline 与共享转发脚本；Fission 入口独立化。合并后 17 项针对性测试通过，包括默认 20 项任务、单模型参数覆盖、共享数据只准备一次、失败队列退出码以及 Fission 的 50 万池和 30/60 epoch 默认值。

再按 `run_pipeline.sh` 改为 `SEEDS`/`EXPERIMENTS` 显式清单，两入口共 23 项相关测试通过，覆盖多行多 seed、任意行 GPU 分配、同卡任务顺序、失败后继续执行及汇总退出码、重复输出/非法参数预检、Fission 原生默认值及 SIGTERM 子进程清理。未运行正式实验，未改动参考脚本 `run_pipeline.sh`。

2026-10-23 以标准 baseline/biom 和 RTX A6000 为参考更新配置：98 项自动化测试通过（启动器/W&B、四轮数据、标准模型接口、Fission AL），两个脚本 `bash -n` 和差异空白检查通过。默认 Budding FNO/Transformer width192，Fission 默认标准 FNO width128，biom_refiner_plus 仅作注释可选行。CPU 构造参数计数不等于 GPU 峰值测试；未启动正式实验或 GPU 短跑。

2026-10-23 大batch训练计划调整：23项针对性测试通过，核对所有默认策略获得一致的模型配置及对应epoch/warmup，脚本语法和差异检查通过；本次仅修改配置，未启动正式实验。

2026-10-23 存储迁移完成：119,661个普通文件、741个符号链接及全部目录已由rsync传输；最终差异检查为空，124,249个条目路径/大小/mtime_ns/权限/链接目标逐项一致。项目原experiments已改为指向机械盘的兼容链接，已验证的项目盘源副本全部清除。记录位于 `/home/users/zsy/mydata/budding_yeast/experiment_storage_migration_2026-10-23.json`。路径修改的23项测试及训练入口帮助、脚本语法检查通过。

### 2026-10-23 数据预生成与离线监控

正式训练由用户手动启动。先运行 `PREPARE_ONLY=1 bash run_al_budding.sh`（Fission 同理），只准备候选池和初始数据，不启动训练；下次正常启动会验证并复用。默认 Budding 文件为 `budding_pool_1000000_seed42_balanced.npz`、`budding_initial_train10000_seed42_balanced.npz`；Fission 为 `fission_pool_500000_seed20000_al.npz`、`fission_initial_train5000_seed42_al.npz`。分布/参数修改使用各自系统 recipe；Budding pool两个profile均为mutant_aware，按模块集合/合法模式组合两级容量配额分配，与class加权的部分静态采样器不同。

本次机械盘准备日志与状态在 `/home/users/zsy/mydata/budding_yeast/al_preparation_2026-10-23/`：`budding.log`、`fission.log`、`status.json`；两系统完成后分别生成 `budding_audit.json`、`fission_audit.json`。审计入口为 `python -m budding_yeast_v2.scripts.analysis.audit_al_preparation --help`。状态 passed 才表示完成全量审计，running 只表示任务仍在进行。正式训练未启动。

离线监控实测：24 项流程/启动器测试通过，其中直接初始化真实 W&B SDK、显式给父环境 online，并验证 AL logger 仍为离线（Lightning 将 offline 转为 SDK 的 dryrun 别名）、四个 round 的 loss 及最终 AL 指标均存在，且产生 `offline-run-*/run-*.wandb`。检查不连接云端，也不训练正式模型。

### 2026-10-23：候选池多核重生成

用户授权停止旧串行任务并重生成。`data/systems/parallel_pool.py` 将候选提议按固定256条分块，在多进程中调用原采样器；主进程按块顺序执行全局去重和碰撞救援，拒绝样本仍回到同一 L 层补采。Budding 保留参数＋初值的精确重复判断；Fission 保留原 allclose 容差、real 和累计池参考索引。参数修改、强度分布、ODE、初值与标签不变。单 worker 的生产入口保留旧串行实现。

多 worker 新池的 `sampling_rng_contract=pool_seedsequence_chunks256_global_dedup_v1`，seed 由根 seed、块编号和补采轮次派生，与进程完成先后和 worker 数无关。各块独立扰乱 Sobol，而不是沿用旧版单个全局 Sobol 流，因此新文件不会逐行等于旧串行文件。测试比较并行函数在1/2 worker下的520条样本顺序和完整 overrides，且重新检查全部跨块碰撞与重复。

两个 AL 脚本的 `SIM_WORKERS` 默认32，本次两系统各32进程，总计最多64个 worker；初始 KMeans 的 OpenMP 线程数设为16，BLAS设为1，避免每个仿真进程再扩张线程。全局协调、压缩写盘并非所有时刻都能占满64核。固定启动副本保存在机械盘准备目录，使用 `AL_SCRIPT_ROOT` 指回项目，避免运行中编辑 shell 文件造成读入错位；旧串行 Budding pool 保留为 `budding_pool_1000000_seed42_al_serial_backup.npz`，不进入本次训练。

已完成实际 CLI 检查：Budding 1000条、Fission 20条（含原生预热）均成功导出；正式百万/50万重生成和初始集仅在 PREPARE_ONLY 模式运行，仍不启动正式训练。

并行重启前后验证：87项回归测试通过（包含跨worker可复现、全局碰撞/唯一性、四轮流程、启动器离线W&B、精确三模块及救援规则）。旧保护清单对 Fission al_pool.py 的字节相等约束因本次授权的并行编排变更而排除，Fission 生物学文件和 data 根目录原始生成器仍继续受保护。重启后实测两系统64个worker，进程CPU合计约3400%（占用随阶段变化），两套日志均持续输出 accepted 数量。

### 2026-10-23：tmux 后台正式实验授权

用户授权在 neuralop 环境中后台启动 Budding，并明确要求 GPU0–3 空闲后再运行。会话为 `tmux attach -t al_budding`，control/data 两窗口；启动状态保存在机械盘 `experiments/al_budding/overnight_2026-10-23/status.json`，控制台为 control.log，批调度输出为 batch.log。正式启动前等待 Budding 准备任务 passed，再单独复查完整输入；Fission 准备任务独立继续，不启动 Fission 正式实验。

新增启动选项：`SKIP_PREPARE=1` 要求 pool/初始文件已经存在，仅运行原有兼容性验证，缺失时直接报错，禁止回退生成；`WAIT_FOR_GPU=1` 在每张卡的每个实验前检查该卡显存，默认占用不超过512MiB才开始，30秒重查一次。未启用 WAIT_FOR_GPU 的历史调用行为不变。测试覆盖缺失输入拒绝、已有输入复用及忙卡等待后启动。

tmux 控制进程激活 `/home/users/zsy/.conda/envs/neuralop`，W&B 强制离线；数据通过校验后使用 `run_al_budding.sh` 的固定副本启动，防止运行中的 shell 被编辑导致语法错位，AL_SCRIPT_ROOT 仍指向真实项目。调度保持四模型×四策略、同卡串行、跨卡并行，已有正式实验目录不覆盖。后台等待状态不等于已经训练，需结合具体 train.log、rounds/round_00/training_history.csv 和 checkpoint 确认训练进展。

同一 tmux 会话的 health 窗口每30秒只读检查 config、pool_manifest、round数据/权重/曲线、本地 offline W&B 和异常日志，写入 overnight_2026-10-23/health.json；缺少尚未生成的轮次文件不被误报成失败。该检查不修改模型和实验数据，不替代未来未知错误的人工诊断。

### Budding 全量数据完成与正式放行

Budding pool 全量审计通过：1,000,000候选，L1/L2/L3=50,000/400,000/550,000；初始合成12,500，L1/L2/L3=625/5,000/6,875；固定 train/val/test=10,000/1,250/1,250，另126株 real。初始仿真失败补采数为0，有限负值保留，两个 profile 均为 mutant_aware。报告位于 `al_preparation_2026-10-23/budding_audit.json`，正式入口另做复查并保存 overnight目录的 inputs_audit.json。

用户后续明确取消 Fission：其专属进程组已终止，准备状态标记 cancelled，Budding保留。共用旧准备监控停止后，已补齐 Budding 完成状态并释放 tmux 门控。GPU0–3空闲后四个正式 FNO 任务已启动（Random/Uncertainty/Power/BAIT），入口使用 SKIP_PREPARE，12626样本和固定切分读取成功，日志确认 W&B offline；大候选池载入阶段只占CPU，不能把配置文件已出现误报为已完成首个训练epoch。正式运行状态以 train.log 与 GPU/health窗口为准。

正式启动验证已通过：四个 FNO 实验均完成至少14个epoch（核查时14/17/14/16），loss正常下降且无Traceback。实际 round_00/dataset.npz 中 train/val/test/real 的名称集合分别与共享初始文件一致，L字段共12626行；每个 pool_manifest 的初始训练数10000、池剩余987500，preprocessing.pt和本地offline-run均存在。检查记录为 `experiments/al_budding/overnight_2026-10-23/runtime_start_verification.json`。仅确认正式训练已正常开始，不代表四轮或16项实验全部完成；最终评测完成后再更新排行榜。

2026-10-28并发调度验证：23项测试通过，含同卡重叠/上限/空槽补位、设1恢复串行、失败继续/停止后续但收尾运行项、idle等待不阻塞自有第二槽位、SIGTERM清理全部并发训练子进程、共享数据和W&B本地离线；仅使用模拟任务与临时离线日志，未启动正式训练。shell语法、DRY_RUN及diff空白检查通过。

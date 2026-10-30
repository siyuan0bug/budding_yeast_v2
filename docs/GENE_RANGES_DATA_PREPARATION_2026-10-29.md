# 新范围候选池与共享初始集：生成及幅值审计（2026-10-29）

**已完成百万pool、共享初始集和接口审计。初始train大值比例由19.55%降至11.79%，相对减少39.69%，但大值没有消失。没有按幅值删除样本，也没有新的模型训练精度结论。**

## 正式输入

- pool：`/home/users/zsy/mydata/budding_yeast/budding_pool_1000000_seed42_gene_ranges.npz`，1000000条参数＋float64初值＋mutation/来源元数据，不含预仿真轨迹。
- initial：`/home/users/zsy/mydata/budding_yeast/budding_initial_train10000_seed42_gene_ranges.npz`，12500条synthetic＋126 real，共12626条完整39×500轨迹。固定seed42划分train10000、validation1250、synthetic test1250、real126；real不参与训练或归一化拟合。
- pool按当前模块/等位型容量均衡：L1/L2/L3=9777/126089/864134。初始选点仍按5/40/55在各层的全pool逐维Z-score参数空间分别KMeans，取得625/5000/6875条后随机划分。新train实际L1/L2/L3=500/4002/5498。
- 使用neuralop环境。pool提议和ODE使用48进程；KMeans局部使用32个native线程，选点结束恢复单线程native库，避免48个ODE进程再各开32线程。KMeans三层约1.7/36.1/53.8秒。合成轨迹仿真失败0条，real基准完整。

## CLB2 GAL-MC确已参与配额

完整pool有**13703条GAL-MC候选**，分布于2487个合法完整等位型组合单元。按模块集合等额、集合内合法模式容量均衡分配，并非额外指定全局GAL-MC百分比。

CLB2单模块配额571条，具体为GAL-MC 8、普通GAL90、GAL+db90、KO2、MC21、MC+db90、db90、pLOF90、ts90。GAL-MC的整数2～10共9种输入，仅有mdt150；排除与real #50相同的单模块8拷贝后剩8种，全部覆盖。其他基因背景下的8拷贝仍可合法出现，不能把所有8拷贝一律排除。

初始集有164条synthetic GAL-MC，其中130条进入train。核对了整数拷贝、mdt150、ksb2pp保持0.04及完整metadata。未自动加入GAL-MC+db。

## 幅值是否缓解

以下严格比较**旧/新初始train各10000条**，不是旧Random最终70000条。两组size、选择协议与split seed相同，但候选池、强度范围及模式目录同时改变，不能分离某项调整的独立因果贡献。

旧实际幅值规则：任意非MASS变量峰值>max(2×126 real逐变量最大值,2)，或MASS>10。

| 范围 | 旧初始集 | 新初始集 |
|---|---:|---:|
| train触发旧规则 | 1955/10000，19.55% | 1179/10000，11.79% |
| 全部synthetic触发旧规则 | 2436/12500，19.488% | 1464/12500，11.712% |
| validation触发旧规则 | 240/1250，19.20% | 135/1250，10.80% |
| synthetic test触发旧规则 | 241/1250，19.28% | 150/1250，12.00% |
| MASS>10 | 0 | 0 |

train减少776条，即7.76个百分点、相对39.69%。如果改用字面“任意变量>2×real最大值、不设置floor2”，则为22.37%→14.97%，不可和上表混用。

| 变量 | 旧P99 | 新P99 | 旧最大值 | 新最大值 |
|---|---:|---:|---:|---:|
| CLN2 | 85.40 | 21.34 | 199.68 | 71.34 |
| CLB2 | 196.63 | 92.62 | 582.63 | 389.54 |
| SIC1 | 75.05 | 60.99 | 430.93 | 399.28 |
| CDC6 | 51.08 | 29.44 | 279.20 | 142.93 |
| ORI | 2637.05 | 1247.17 | 7573.95 | 5255.58 |
| BUD | 77.52 | 24.07 | 264.38 | 84.34 |

log1p后的训练标准差也有所降低，例如CLN2 0.7565→0.6486、CLB2 0.7747→0.7054、CDC6 0.8102→0.7299、ORI 1.7504→1.6824；这仅描述尺度变化，不等同于FNO精度改善。

剩余极值仍有具体组合来源：CLB2最大值来自CDC20-ts＋CLB2-GAL-MC10（ksb2p1.2），同时给出ORI最大值；SIC1最大值来自CLN2-ts＋SIC1-MC16+PM＋SWI5-GAL。说明按基因限幅后，表达、稳定性与失活的组合仍可造成强蓄积。没有据此恢复变量阈值过滤。

## 数值与数据契约

- `audit_al_preparation.py`全量检查通过：pool/初始来源、固定切分、参数/变量顺序、float64初值、元数据、L、标签、real完整性及初始轨迹有限性。
- 当前新12500条synthetic没有TEM1/CDC15/ESP1超过各自总量上界（容差1e-3×max(1,total)）；旧初始集有2条TEM1越界。本轮没有修改旧数据。
- 有限微小负值按用户要求保留，新train最小约−7.74e-7；没有静默裁剪原始轨迹。
- 另外对新CLN2/CLB2/SIC1/CDC6最大样本执行严格ODE复算，记录在`strict_peak_rechecks.json`；是诊断检查，不替换正式文件。
- 新旧real峰值一致；旧pool和initial的size/mtime保持，生成源码哈希固定。完整生成源码快照和manifest随审计保存。

## 重新运行实验

`run_al_budding.sh`默认已改用`DATA_COHORT=gene_ranges`，以新pool、新initial和新的结果子目录配对。23项launcher/本地W&B日志测试通过，dry-run确认FNO的6个baseline＋ours_response、GPU0～3、每卡最多2任务、batch512、50/50/50/100 epochs，W&B强制offline。没有在本次数据准备中启动正式训练。

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
SKIP_PREPARE=1 MAX_JOBS_PER_GPU=2 bash run_al_budding.sh
```

结果路径：`/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno/<loss_strategy>/w192_n4_m24_b512_e50-100_lr1e-3_wm4-8_al4_refit_gene_ranges/seed_42/`，不会复用旧`refit_balanced`目录。需要等待对应GPU空闲时额外设置`WAIT_FOR_GPU=1`。

## 审计产物

`/home/users/zsy/mydata/budding_yeast/analysis/gene_ranges_preparation_2026-10-29/`包含日志、manifest/source快照、preparation_audit.json、pool_quota_summary.json、initial_gal_mc_counts.json、launcher_dry_run.txt。`comparison/`内有全部39变量分位数/最大值/阈值计数、按L统计、summary.json、极值实例overrides、严格复算以及PNG/PDF比较图。

生产分析入口：`scripts/analysis/dataset_diagnosis/compare_initial_sampling_domains.py --new ... --old ... --pool ... --output ...`。数据准备在tmux会话`budding_data_ranges_20261029`完成，最终`stage.txt=complete`，会话随任务正常结束。仿真产生标签，未仿真的pool不伪造轨迹/Pattern标签。

补充：2026-10-29已再次用当前新pool/initial和真实FNO checkpoint完成7策略dtype复验（128候选切片）；15项dtype回归通过。详见[精度边界修复](AL_DTYPE_FIX_2026-10-28.md#2026-10-29-当前新候选池复验)。不代表新正式训练已经运行。

## 全量real碰撞复核

2026-10-29独立扫描当前1000000条pool与126个canonical real条件，使用现有atol1e-8/rtol1e-5定义，结果0个相同输入；参数前置筛选后已无可能碰撞项。生成报告的collisions=0、rescued=0：有限离散支持在枚举时就跳过real项，因此不进入运行时救援计数；连续/较大支持若发生碰撞才尝试保持模块与环境的救援，仍碰撞则重抽。普通相似输入、同基因型不同强度或近似轨迹不会因“相似”被移走。报告`full_pool_real_collision_audit.json`保存在本审计目录，探针为`scratch/probes/check_full_pool_real_collisions.py`。

# 精确0与连续残余：配对轨迹实验（2026-11-12）

本实验只比较参数设置的动力学后果，不修改生产mutation/分布/求解器，也不开展模型训练或候选池生成。240次生产/严格容差仿真及120次更严格复核全部成功。

## 预先固定的实验设计

8机制×3背景×5残余比例＝120组物理输入。每组在生产BDF（rtol=1e-4、atol=1e-6）、严格BDF（1e-8、1e-10）以及复核BDF（1e-10、1e-12）各算一次。210min/500输出点；所有事件、重置、初值来自同一canonical生产函数，严格求解仅通过进程局部wrapper覆盖容差。生产超时5秒，严格诊断20秒，复核诊断30秒；没有超时失败。

比例alpha为0、0.001、0.005、0.009、0.01；后四者对应当前U(0,0.01)的10%、50%、90%分位点和支持上限。各目标参数按WT×alpha设置，多参数共用alpha。每个固定背景内实际初值和环境相同，唯有目标速率变化。CDC14只测合成关闭分支。APC-A两个模块入口合并为一次机制实验。

WT背景指其他参数/初值保持WT，仍加入目标mutation。额外背景包括合法GAL复合、其他基因KO、MC及NET1部分结合背景；它们是预先手工定义的诊断背景，不是随机估计总体频率。NET1 TAB6遵循生产函数独立抽结合残余的规则（固定RNG=42），真实kasrent/kasrentp值以design.json的overrides为准，不能把传入的占位alpha误作该函数实际使用的残余。

部分完整输入恰好等于canonical real，是刻意保留的诊断对照，已逐条标记complete_input_matches_real；没有加入训练或用于拟合选样分数。

## 定量结果

差异定义为同一背景、同一精度下相对于精确零轨迹的39变量整体RelL2：||Y_alpha-Y_zero||/||Y_zero||。每种机制12个比较（3背景×4非零残余）。下表使用严格BDF结果；包含相位偏移和幅值差，不能直接当作某个变量的相对幅值变化。

| 机制 | RelL2中位数 | 最大RelL2 | 最大差异背景 | 分裂次数或六分类变化次数 |
|---|---:|---:|---|---:|
| CLB5_db | 0.88% | 3.01% | GAL_CLB5_CDH1_KO | 0/12 |
| SIC1_db | 9.15% | 12.93% | WT | 0/12 |
| PDS1_db | 1.83% | 10.53% | WT | 0/12 |
| CDC20_ts | 0.18% | 7.53% | PDS1_KO_CLB5_KO | 0/12 |
| CDC15_ts | 0.34% | 7.74% | NET1_TAB6 | 0/12 |
| APC_Core_ts | 6.88% | 13.26% | MC_SIC1 | 0/12 |
| CDC14_ts_synthesis | 0.13% | 0.50% | GAL_SIC1 | 0/12 |
| APC_A | 2.52% | 10.31% | WT | 0/12 |

96个配对比较中，分裂事件次数、ABC/六分类和生物诊断标签均未改变。它说明这些预设背景在210分钟内没有观察到上述分类切换；不证明其他背景或更长时间里不发生切换，也不证明自持极限环/真正分岔是否相同。

## 可见差异的例子

- APC_Core-ts＋7拷贝SIC1：零到alpha=0.01的整体RelL2为13.26%，最大绝对差发生在ORI，约19.42；两者均观察到1次分裂/P2。
- SIC1-db，其他条件WT：整体RelL2最高12.93%，SIC1最大绝对差约1.616；210min时SIC1由精确零下的1.982965变为alpha=0.01时的0.367331，终值下降约81.48%；两者均无分裂/P5a。全变量聚合RelL2会稀释个别变量的明显差异。
- PDS1-db，其他条件WT：整体RelL2最高10.53%，首次分裂由约191.285min提前至188.661min；两者仍为1次分裂。
- CDC14合成关闭分支：三个背景所有比较的最大整体RelL2仅0.502%，在本窗口内几乎重合。

因此近零与零不是对每种机制都完全等价，但本实验也不支持一概声称这点残余会导致Pattern跨类。是否给精确零分配概率，还需结合突变机制定义；不能只用这24个背景中的最大差异作全池概率。

## 数值误差复核

生产默认容差对严格结果的整体RelL2中位数为0.0283%，最大为8.39%。少数尖锐事件轨迹对默认容差敏感；这也是图中采用严格结果的原因。没有修改正式生产容差。

进一步从1e-8/1e-10收紧到1e-10/1e-12，120例全部成功，最大整体差异为0.00340%，中位数为0.00067%，分裂次数与六分类无不一致。因此上述约10%级的轨迹差异并非本次两档严格容差之间的数值差异。

轨迹保留有限负值；原始物理RelL2不截零。log1p辅助指标仅在计算时将微小负值截零，原轨迹未修改。

## 曲线与原始数据

目录：`/home/users/zsy/mydata/budding_yeast/analysis/zero_vs_residual_20261112/`。

- `key_examples.png`：SIC1浓度变化、PDS1分裂时刻放大及CDC14近重合三种情形。
- `largest_differences.png`：三种不同机制各自差异最大的背景，显示MASS、CLB2及最大绝对差变量；明确为选出的例子。
- `all_mechanisms_WT.png`：全部8机制的WT背景总览，没有按差异筛选。
- `all_24_contexts.pdf`：全部24背景，每页9个变量、5条曲线。
- `plots/<mechanism>__<context>.png`：每组独立曲线图。
- `comparisons.csv`：192个生产/严格配对指标；`numerical_checks.csv`和`ultra_verification.csv`用于误差核对。
- `design.json`、`runs.json`、`trajectories.npz`、`ultra_runs.json`、`ultra_trajectories.npz`和`experiment_manifest.json`保存输入、轨迹、事件时间、标签及源码哈希。

## 复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/compare_zero_residual_trajectories.py \
  --workers 8 --output /home/users/zsy/mydata/budding_yeast/analysis/zero_vs_residual_20261112

# 复核更严格容差；复用已存输入，不重跑前两档
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/compare_zero_residual_trajectories.py \
  --verify-ultra --workers 8 --output /home/users/zsy/mydata/budding_yeast/analysis/zero_vs_residual_20261112

# 只重画已保存轨迹
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/compare_zero_residual_trajectories.py \
  --plot-only --output /home/users/zsy/mydata/budding_yeast/analysis/zero_vs_residual_20261112
```

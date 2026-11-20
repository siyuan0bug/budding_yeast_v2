# 全部126个real-mutant配置的采样精度审计（2026-11-11）

**结论：400点作为当前效率与波形保真的折中仍有依据，但短暂的初始快速弛豫和跳变边缘存在可见局部差异；不能声称对训练精度完全没有影响。**

## 覆盖与口径

- 全部126个canonical real-mutant配置（包括WT及不同培养条件），126个成功，失败0个；不依靠四个示例概括全体。
- 同一次生产BDF积分（rtol=1e-4、atol=1e-6），按原有native线性插值分别直接输出300/350/400/500点，再线性重构到0.01min、21001点网格。500点也作为候选，不作为零误差真值。
- 参考是相同容差的数值仿真，不是解析真解、更严容差收敛结果或实测生物时间序列。此次没有训练/评测新神经网络。
- 全39变量log相对L2按每株计算：`||log1p(max(recon,0))-log1p(max(ref,0))||₂ / ||log1p(max(ref,0))||₂`。
- CORE9使用MASS、CLN2、CLB2、CLB5、SIC1、CDC20、CDH1、ORI、SPN。物理误差逐株逐通道除以`max(该参考轨迹峰峰值,0.1)`，避免近常量通道分母接近零。
- 平滑区排除任一event前后1min，#104另排除180min系数切换前后1min；初始快速弛豫仍保留。event邻域单独报告，仅在有event/系数切换的配置上汇总。
- 最高峰损失指每个CORE9通道整个210min内的参考最大值减采样最大值，除以上述幅度尺度，再取每株最坏通道；不是每一个次级峰的逐峰幅值误差。
- 显著峰数量只检查六个连续核心通道CLN2/CLB2/CLB5/SIC1/CDC20/CDH1，动态幅度小于0.1时跳过。prominence=max(幅度10%,0.01)，参考网格最小峰间隔1min，采样网格按最近整数点间距近似。该启发式不检查MASS/ORI/SPN重置次数或微小振荡。

## 全体结果

| 点数 | dt/min | 全39变量log相对L2均值 / P95 | 平滑CORE9 NRMSE均值 / P95 | 最坏最高峰损失 | 显著峰数改变配置数 |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.7023 | 2.544% / 4.604% | 0.101% / 0.425% | 0.957% | 0/126 |
| 350 | 0.6017 | 2.375% / 4.533% | 0.077% / 0.314% | 0.817% | 0/126 |
| 400 | 0.5263 | 2.225% / 4.118% | 0.060% / 0.238% | 0.612% | 0/126 |
| 500 | 0.4208 | 1.994% / 3.700% | 0.039% / 0.149% | 0.466% | 0/126 |

所有误差均为相对细网格的采样重构误差，不是FNO预测误差。四档各株的连续核心通道显著峰数均无改变；这不等于所有局部细节或所有39通道均完全相同。

| 点数 | CORE9全程幅度归一化MAE均值 | CORE9 event邻域NRMSE均值 / P95 | CORE9时间点中误差>幅度5%的比例（跨株均值） |
|---:|---:|---:|---:|
| 300 | 0.0336% | 3.96% / 8.02% | 0.0807% |
| 350 | 0.0282% | 3.76% / 7.60% | 0.0687% |
| 400 | 0.0228% | 3.34% / 6.79% | 0.0580% |
| 500 | 0.0174% | 2.96% / 6.14% | 0.0442% |

较小的全程MAE不能掩盖短暂局部误差；跳变附近的误差由持续时间短但幅度大的偏差贡献。某些株400点的log相对L2甚至比300点大，原因是网格与跳变时刻对齐不同，不代表加密总体更差。

## 敏感配置与图像

- #118 APC-A cdh1Δ multi-copy SIC1：CLB2在起始不到1min快速衰减，较粗输出将它画成更宽的下降斜坡；500点也未完整分辨全部亚分钟细节。400点时该通道平滑区整段NRMSE约1.27%，500点约0.84%，是这次最敏感的核心平滑通道。
- #110 TAB6-1 clb5Δ clb6Δ、#100 GAL-CDC14、#30 GAL-SIC1 cln1Δ cln2Δ：CDH1初始快速升高的亚分钟曲率有差异，210min全图上差异较小。
- #122 APC-A cdh1Δ multi-copy CDC20：300点的CLB2通道最高峰损失最明显，为幅度约0.957%；主要周期波形及显著峰数保持。
- #36 sic1Δ cdh1Δ、#95 cdc15-ts net1-ts、#109 TAB6-1 cdc15-ts：MASS分裂跳变的斜坡宽度与表观位置变化，放大局部后可见；不等于积分器的真实event触发时间改变。

![敏感配置全程与局部图](../../scripts/analysis/outputs/time_resolution_2026-11-11/all_real/sensitive_examples.png)

[八组敏感通道全程与局部放大](../../scripts/analysis/outputs/time_resolution_2026-11-11/all_real/worst_cases.png)；[126配置误差分布](../../scripts/analysis/outputs/time_resolution_2026-11-11/all_real/distributions.png)；[全部126页CORE9曲线PDF](../../scripts/analysis/outputs/time_resolution_2026-11-11/all_real/all_real_mutants.pdf)；[逐株逐档原始JSON](../../scripts/analysis/outputs/time_resolution_2026-11-11/all_real/audit.json)。

## 选择与实验影响

用户关心整体可见波形和模型实验结果，不强求区分图中不可见的相邻event。按此目标，400点（dt≈0.5263min）仍是合理候选：其全39变量log重构误差均值为2.225%，500点为1.994%，相差0.231个百分点；CORE9最高峰最大损失为幅度0.612%。300点也未改变显著峰数，但初始快速变化和跳变边缘更粗。

这比只观察WT和三个示例更有依据，但没有证明400点训练后的FNO与500点精度等价。此次126配置用于分辨率开发诊断，不能再声称是完全未接触的盲测；后续模型比较应保留共同的细网格评测，并至少重复多个训练seed。此前本轮A6000/FNO的400点更新耗时约减少27%、allocated显存约减少17%，仅沿用计时，不将它解释为完整AL实验同比提速。

生产求解器、原始数据、标签、AL网格与默认值均未改变。

## 复现

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle \
/home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/audit_real_mutant_time_resolution.py \
  --all-pages --output scripts/analysis/outputs/time_resolution_2026-11-11/all_real
```

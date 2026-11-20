# Mutant-aware精确关闭规则：已实施及修改前后对照

本次已获得用户执行授权。只落实已讨论的关闭通道；CDC14-ts按用户最新约束整体不变。没有修改canonical参数、39维状态/初值、ODE、事件、real定义、Fission或data根目录原始生成文件，也没有启动正式训练或生成新的大型候选池。

## 最终生效参数的分布对照

alpha为修改前U(0,0.01)的残余比例，同一个等位干预内多参数共用alpha；APC-A原先是在mutation赋值函数内部抽这个残余，而非其标量分布接口。表中比较的是最终物理参数，而不只比较标量返回值。

| 机制 | 修改前MA的生效参数分布 | 修改后 |
|---|---|---|
| CLB5-db | kdb5pp=0.16alpha，即U(0,0.0016) | kdb5pp=0 |
| SIC1-db | kd3c1=alpha，即U(0,0.01) | kd3c1=0 |
| PDS1-db | kd2pdspp=0.2alpha、kd3pdspp=0.04alpha | 两项同时0 |
| CDC20-ts | ks20p=0.006alpha、ks20pp=0.6alpha | 两项同时0 |
| CDC15-ts | kkpnetpp=0.6alpha，即U(0,0.006) | kkpnetpp=0 |
| APC_Core-ts | ks20p=0.006alpha、ks20pp=0.6alpha、kscdh=0.01alpha | 三项同时0；CDH1T/CDH1初值继续0 |
| APC-A（CDC20/APC_Core两个入口） | ka20pp=0.2alpha，即U(0,0.002) | ka20pp=0 |

合计7类机制、8个gene/mode入口、12个参数分量（涉及9个不同物理参数）从连续残余变为确定性的零原子。随机接口与逆CDF接口均返回0；mutation赋值函数也独立强制这些关闭，防止专用采样器自行传非零alpha后恢复残余。

### 明确保留

- CDC14-ts原50/50两分支逐项保留：合成分支ks14~U(0,0.002)；活性分支ks14=0.2且beta~U(0.05,0.35)，四项活性=(4beta,4beta,2beta,0.8beta)。它与#99等real的ks14=0仍不同，这是用户明确保留的当前设计。
- CLB2-db仍为kdb2P=0、kdb2pp~LogUniform(0.017,0.068)。
- TEM1/ESP1/NET1-ts及TAB6-1的非零规则保持；ESP1的论文数值差异没有在本次修改中处理。
- MC、GAL、GAL-MC、DAmP、pLOF及其他未涉及模式的分布保持；MASS的分布、赋值和历史逆CDF标量行为保持。没有把整个MA切成MASS。
- 只关闭指定速率，不额外清零蛋白初值。GAL+db继续叠加GAL合成修改；APC_Core-ts已有的CDH1初值关闭继续生效。

## 下游同步

1. `distributions/ranges.py`集中保存`EXACT_ZERO_TARGETS`；随机分布、MA逆CDF、mutation赋值及容量判定使用同一清单。共享逆CDF给MASS提供旧行为兼容分支，避免此次修改改变MASS标量接口。
2. WT-blind旧ledger代码会在调用mutation后覆盖部分残余。已为固定关闭模式增加独立处理分支，避免重新写成非零。
3. 候选校验拒绝与声明的固定关闭模式不符的非零参数；有效real行仍使用原有豁免。
4. `quotas`和`al_pool`把新固定模式识别为有限容量，而不是无限连续空间。小有限支持按有效输入签名枚举，排除完整real输入，并保留跨模式去重。例如相同背景下CDC20-KO/ts现在具有相同有效输入，不能因为模式名字不同重复计数。
5. 碰撞救援不加epsilon、不改mdt。70%弱化KO分支若先选到与KO等效的CDC20-ts，会继续尝试该基因其他合法弱模式（如DAmP），而不是错失可用的弱化救援后直接落入其他模式。
6. AL响应/敏感性特征从实际规则推导，已验证这些固定模式不再产生假的连续强度坐标。
7. 范围指纹及依赖它的catalog版本已变，旧候选池不能静默作为新规则缓存复用。原有数据文件不被修改；新规则下必须重新生成需要使用的pool/初始集。同seed的完整样本序列也可能因固定分支不再消耗随机数而改变。

## LHS边际配置

已重新编译到`/home/users/zsy/mydata/budding_yeast/analysis/offline_pool_plan_20261112_exact_zero/`：

- `before_after_distributions.csv`：12个生效参数分量逐项修改前后对照。
- `mixed_lhs_domain.json`及`selected_mixed_lhs_components.csv`：这些模式的连续残余分量变为point(0)，权重仍按相同pre-rejection随机mutation提案计算。
- `exact_zero_validation.json`：90个未覆盖参数的0.5–1.5WT窗口不变、全部边际权重和为1、CDC14及其他无关边际完全不变。
- `remaining_residual_audit/`：原9个zero/residual入口目前只剩用户明确保留的CDC14合成分支。
- `parameter_evidence/`：38项模块参数见证、128项模式足迹及12项复合完整输入检查完成；固定db现在具有真正零原子。

这里完成的是边际配置同步，不是宣称联合LHS生产采样器或新两类10万完整轨迹池已经完成。90个参数未新增统一零点概率。去重/碰撞救援/仿真拒绝后的经验频率仍可能与pre-rejection提案不同。

## 验证及保护

本次编辑前保存了独立源文件和执行结果快照，未用git HEAD覆盖用户已有改动：

- 366条未涉及模式/MASS的随机抽样、赋值和RNG状态与快照一致；122个模式接口的5点逆CDF结果一致。
- CDC14另外用100个seed对比旧函数，覆盖合成/活性两个分支，参数结果及RNG状态均完全一致。
- 8个固定入口的随机/逆CDF/任意alpha直接赋值、真实初值、WT-blind保护、候选拒绝、有限容量和AL强度特征均有回归检查。
- data根目录原始代码、canonical生物定义、Fission及MASS等63个保护文件的SHA256与本次开始时完全相同。
- 120项相关回归全部通过（79项范围/救援/新关闭规则/元数据，41项各采样器精确阶数/参数池导出）；AL强度特征优化后又单独复核2项通过。2026-10-22的旧保护manifest已过时，相关单项明确排除，以本次独立63文件快照保护代替，不修改旧manifest掩盖差异。

快照：`scratch/probes/exact_zero_update_20261112/`（忽略目录）。测试：`tests/test_canonical_exact_zero_update.py`等。没有据此声称训练精度必然改善；生物机制依据、ESP1数值差异与配对实验限制仍见原讨论和实测文档。

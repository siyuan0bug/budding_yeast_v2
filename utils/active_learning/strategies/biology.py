"""strategies biology services extracted from the pre-package implementation."""
from budding_yeast_v2.data.systems.budding.api import BINDING_GROUPS as MASS_BINDING_GROUPS
from budding_yeast_v2.utils.active_learning.pool import _mass_desc_of
from budding_yeast_v2.utils.active_learning.pool import _mass_sample_genes
from budding_yeast_v2.utils.active_learning.pool import _parse_mass_gene_modes
import numpy as np


def _maal_sample(self, trainer, pl_module, dm, device, current_epoch):
    """
    MAAL (Mutant-Aware Active Learning): diversity + target priority 加权。

    acquisition_score = (1 - target_weight) * diversity + target_weight * target_priority
    - diversity: 候选样本到训练集的最小欧氏距离 (轨迹形状感知特征: MASS pattern + 全变量统计 + 参数)
    - target_priority: 完全基于训练集计算 (零数据泄露)
        = 0.5*同家族MC dropout不确定性 + 0.3*候选到训练集同家族参数距离 + 0.2*家族覆盖度
      不再访问 dm.test_dataset_real, 严格遵守数据隔离原则。

    权重通过 --al_target_weight 控制 (默认 0.4), diversity 权重 = 1 - target_weight。
    """
    if self._pool_size == 0:
        print("[MAAL] 候选池已空, 跳过本轮 AL")
        return None

    # Step 1: 对候选池和训练集做 forward, 得到轨迹形状感知特征
    print(f"[MAAL] Step 1: 计算轨迹形状感知特征 (池: {self._pool_size})...")
    pool_features = self._compute_pool_features_shape_aware(pl_module, dm, device)
    train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
    # 用 train 统计量分组 z-score 标准化
    pool_features, train_features = self._standardize_shape_features(pool_features, train_features)

    # Step 2: 计算 diversity score (候选到训练集的最小距离)
    print("[MAAL] Step 2: 计算 diversity score (min distance to train)...")
    diversity_scores = self._compute_diversity_scores(pool_features, train_features)
    diversity_scores = self._normalize_array(diversity_scores)

    # Step 3: 计算 target_priority (完全基于训练集: 同家族MC不确定性 + 参数距离 + 覆盖度)
    # 数据隔离: 不访问 dm.test_dataset_real, 零数据泄露
    print("[MAAL] Step 3: 计算 target_priority (训练集版: 同家族MC不确定性 + 参数距离 + 覆盖度)...")
    family_uncertainty = self._compute_train_family_uncertainty(pl_module, dm, device)
    target_priorities = self._map_family_priority_train_based(family_uncertainty, dm)
    target_priorities = self._normalize_array(target_priorities)

    # Step 4: 加权 acquisition
    div_w = 1.0 - self.target_weight
    tgt_w = self.target_weight
    acquisition = div_w * diversity_scores + tgt_w * target_priorities
    print(f"[MAAL] Step 4: 加权 acquisition (diversity={div_w:.2f}, target={tgt_w:.2f}) | "
          f"score range: {acquisition.min():.4f}~{acquisition.max():.4f}")

    # Step 5: top_k 选取
    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    sel_idx = np.argsort(acquisition)[-n_select:]
    print(f"[MAAL] Step 5: top_k 选取 {n_select} 个样本")
    return self.make_selection_plan(sel_idx, dm)


def _al4pde_mass_sample(self, trainer, pl_module, dm, device, current_epoch):
    """
    al4pde MASS: 借鉴 data/mass_dataset_generator.py 的"通路→机制→严重程度"分层采样思想,
    将其从"数据生成阶段"迁移到"AL 选择阶段": 优先选择生物学上高频、机制典型的候选样本。

    acquisition = (1 - mass_weight) * diversity + mass_weight * mass_priority
    - diversity: 候选到训练集的最小欧氏距离 (轨迹形状感知特征: MASS pattern + 全变量统计 + 参数),
      防止选样冗余
    - mass_priority: 生物学分层优先级 [0,1], 每个候选按其扰动参数的"通路频率 × 机制频率"打分
        * 通路频率: 参数所属绑定组对应通路的 high/medium/low 频率 (6/3/1), 取参数最高频通路
        * 机制频率: 由候选参数相对 WT 默认值的扰动模式推断机制类 (loss/gain/partial),
          权重取自 Chen 2004 操作实例统计 (KO/GAL-OE/ts)
      越高 → 越像生物学高频、机制典型的突变 → 越优先选择

    权重通过 --al_mass_weight 控制 (默认 0.5), diversity 权重 = 1 - mass_weight。
    数据隔离: 仅用 _pool_P_real / _pool_mutant_names (候选池自带) + 静态生物学先验表, 不访问测试集。
    """
    if self._pool_size == 0:
        print("[al4pde_mass] 候选池已空, 跳过本轮 AL")
        return None

    # Step 1: 计算轨迹形状感知特征 (与 diversity/maal 一致)
    print(f"[al4pde_mass] Step 1: 计算轨迹形状感知特征 (池: {self._pool_size})...")
    pool_features = self._compute_pool_features_shape_aware(pl_module, dm, device)
    train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
    pool_features, train_features = self._standardize_shape_features(pool_features, train_features)

    # Step 2: 计算 diversity score (候选到训练集的最小距离)
    print("[al4pde_mass] Step 2: 计算 diversity score (min distance to train)...")
    diversity_scores = self._compute_diversity_scores(pool_features, train_features)
    diversity_scores = self._normalize_array(diversity_scores)

    # Step 3: 计算 MASS 生物学分层优先级 (通路频率 × 机制频率)
    print("[al4pde_mass] Step 3: 计算 MASS 生物学分层优先级 (通路频率 × 机制频率)...")
    mass_scores = self._compute_mass_priority_scores()
    mass_scores = self._normalize_array(mass_scores)

    # Step 4: 加权 acquisition
    div_w = 1.0 - self.mass_weight
    mass_w = self.mass_weight
    acquisition = div_w * diversity_scores + mass_w * mass_scores
    print(f"[al4pde_mass] Step 4: 加权 acquisition (diversity={div_w:.2f}, mass={mass_w:.2f}) | "
          f"score range: {acquisition.min():.4f}~{acquisition.max():.4f}")

    # Step 5: top_k 选取
    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    sel_idx = np.argsort(acquisition)[-n_select:]
    print(f"[al4pde_mass] Step 5: top_k 选取 {n_select} 个样本")
    return self.make_selection_plan(sel_idx, dm)


def _compute_mass_priority_scores(self):
    """计算候选池每个样本的 MASS 生物学分层优先级 (真实机制元数据版)。

    MASS 候选池在采样时已记录每个样本的真实 (gene, mode) 元数据
    (self._pool_mass_meta, 由 _parse_mass_gene_modes 从 desc_str 解析),
    直接查 mass_dataset_generator.BINDING_GROUPS 的生成先验打分:
      score = mean over (gene, mode) 对 of [ P(group) * P(mode|group) ]
      - P(group):    BINDING_GROUPS[gid]['P_bio_G']  (通路频率, 已归一化)
      - P(mode|group): BINDING_GROUPS[gid]['modes'][mode] (机制频率)
    高频通路 + 高频机制 → 高分, 与数据集生成先验一致。

    对比旧版 (从参数值相对 WT 默认值偏移推断 loss/gain/partial 粗机制):
    无法区分 db/PM/CA/APC-A/TAB6-1/CSL 等结构化模式, 且对 init_*/离散开关
    参数 (mdt/Tem1T/B0) 误判; 真实元数据版完全无此问题。

    数据隔离: 仅访问采样元数据 + 静态先验表, 不访问测试集。
    """
    # gene → group 映射 (BINDING_GROUPS 的 name 字段, G1-G27 无重名)
    gene_to_group = {}
    for g_info in MASS_BINDING_GROUPS.values():
        gene_to_group[g_info['name']] = g_info
    max_p_g = max((g['P_bio_G'] for g in MASS_BINDING_GROUPS.values()), default=1.0)

    # 救援基序中未登记进 modes 的 mode (如 ANTAGONISTIC_TUG 的 'pDEL') 给低默认概率
    DEFAULT_MECH_P = 0.02

    scores = np.zeros(self._pool_size)
    n_empty = 0
    for i in range(self._pool_size):
        meta = self._pool_mass_meta[i] if i < len(self._pool_mass_meta) else None
        pairs = meta.get('genes', []) if meta else []
        if not pairs:
            n_empty += 1          # 纯碳源样本 (无突变基因) → score 0
            continue
        contribs = []
        for gene, mode in pairs:
            g_info = gene_to_group.get(gene)
            if g_info is None:
                contribs.append(0.01)
                continue
            mech_p = g_info['modes'].get(mode, DEFAULT_MECH_P)
            p_g = g_info['P_bio_G'] / max_p_g     # 通路频率归一到 [0, 1]
            contribs.append(p_g * mech_p)
        scores[i] = float(np.mean(contribs))

    n_scored = self._pool_size - n_empty
    print(f"[al4pde_mass] mass priority (真实机制元数据): scored {n_scored}/"
          f"{self._pool_size} 候选 | range: {scores.min():.4f}~{scores.max():.4f}, "
          f"mean={scores.mean():.4f}")
    return scores


def _al4pde_novelty_sample(self, trainer, pl_module, dm, device, current_epoch):
    """al4pde_novelty: 从纯随机候选池中主动挖掘"定向采样等效价值"样本

    研究背景: 定向采样 (rescue motif) 训练集让模型在 real mutants 测试集上
    表现更好, 因其覆盖了 "驱动突变+抑制突变" 的上位性组合。本策略在完全
    随机的候选池上 (无任何定向样本), 通过三个零泄露信号逼近该价值:

    acquisition = w_n * novelty + w_f * family_unc + w_d * diversity
    - novelty (组合新颖性): 候选的突变单元上位性组合在训练集中的覆盖稀缺度
      (单元=(基因,突变类型); 全子集评估: 单元/对/三元/.../完整组合, 尺寸加权
      w_s=2^-|s-2|, 详见 _compute_combination_novelty_scores),
      优先补齐训练集未覆盖的组合 (rescue motif 价值的结构化代理)
    - family_unc (家族不确定性): 训练集 MC dropout 按首基因家族分组的
      平均不确定性, 模型不自信的家族优先补充 (hard family 信号)
    - diversity (轨迹多样性): ODE 真实轨迹形状特征到训练集的最小距离,
      保证参数空间 + mass pattern 双重覆盖 (参数距离与 pattern 一起考量)

    数据隔离 (零泄露): 三个信号仅使用 dm.train_dataset_synth (当前训练集) 与
    self._pool_* (随机候选池自带元数据), 不访问 real mutants 的参数/轨迹/
    性能, 训练过程对 126 个真实突变体完全盲。

    权重: --al_novelty_weight (默认 0.6, novelty 是 rescue motif 价值的主要
    结构化代理, 主导选样) + --al_family_weight (默认 0.2),
    diversity 权重 = 1 - novelty_weight - family_weight 自动归一化。
    """
    if self._pool_size == 0:
        print("[al4pde_novelty] 候选池已空, 跳过本轮 AL")
        return None

    # Step 1: 组合新颖性 (候选 vs 训练集的突变单元全子集覆盖稀缺度)
    print(f"[al4pde_novelty] Step 1: 计算组合新颖性 (单元级全子集覆盖, 池: {self._pool_size})...")
    novelty_scores = self._compute_combination_novelty_scores(dm)
    novelty_scores = self._normalize_array(novelty_scores)

    # Step 2: 家族不确定性 (训练集 MC dropout → 家族分组 → 映射到候选)
    print("[al4pde_novelty] Step 2: 计算家族不确定性 (训练集 MC dropout)...")
    family_unc = self._compute_train_family_uncertainty(pl_module, dm, device)
    family_scores = self._map_family_unc_to_pool(family_unc)
    family_scores = self._normalize_array(family_scores)

    # Step 3: 轨迹多样性 (ODE 真实轨迹形状特征 min-dist, 与 diversity 策略一致)
    print("[al4pde_novelty] Step 3: 计算轨迹多样性 (ODE 形状特征 min-dist)...")
    pool_features = self._compute_pool_features_shape_aware(pl_module, dm, device)
    train_features = self._compute_train_features_shape_aware(pl_module, dm, device)
    pool_features, train_features = self._standardize_shape_features(pool_features, train_features)
    diversity_scores = self._compute_diversity_scores(pool_features, train_features)
    diversity_scores = self._normalize_array(diversity_scores)

    # Step 4: 加权 acquisition (权重自动归一化: w_d = 1 - w_n - w_f)
    w_n = float(np.clip(self.novelty_weight, 0.0, 1.0))
    w_f = float(np.clip(self.family_weight, 0.0, 1.0 - w_n))
    w_d = max(0.0, 1.0 - w_n - w_f)
    acquisition = w_n * novelty_scores + w_f * family_scores + w_d * diversity_scores
    print(f"[al4pde_novelty] Step 4: acquisition = {w_n:.2f}*novelty + {w_f:.2f}*family "
          f"+ {w_d:.2f}*diversity | range: {acquisition.min():.4f}~{acquisition.max():.4f}")

    # Step 5: top_k 选取
    n_select = min(self._al4pde_scheduled_num_add(), self._pool_size)
    sel_idx = np.argsort(acquisition)[-n_select:]
    print(f"[al4pde_novelty] Step 5: top_k 选取 {n_select} 个样本")
    return self.make_selection_plan(sel_idx, dm)


def _compute_combination_novelty_scores(self, dm):
    """计算候选池每个样本的组合新颖性 (v2: 突变单元级全子集覆盖稀缺度)

    生物学动机: rescue motif 定向采样的价值在于覆盖 "驱动突变+抑制突变"
    的特定上位性组合; real mutants 大多为这类多基因上位性组合。本函数在
    不访问 real mutants 的前提下, 度量每个随机候选能带来的"新组合"。

    v2 升级 (相对 v1 "基因对覆盖" 的三点改进):
    1. 突变单元细化为 (基因, 突变类型): CLB2[KO] 与 CLB2[GAL-OE] 是不同
       单元, 覆盖判断按单元进行 —— 不再按基因名合并, 避免训练集见过
       CLB2[KO] 就把候选的 CLB2[GAL-OE] 误判为已覆盖
    2. 组合空间从仅基因对扩展到全子集: 对候选的 k 个单元, 评估
       s=1..k 全部尺寸的组合 (单元/对/三元/.../完整组合), 覆盖完整
       上位性组合结构
    3. 尺寸加权 w_s = 2^{-|s-2|}: 二元上位性 (s=2) 是 rescue motif 价值
       核心取权重 1; 单突变类型 (s=1) 与三元组 (s=3) 取 0.5, 更高阶逐级
       减半 —— 防止必然未见的高阶组合饱和分数、挤掉低阶判别信号

    公式: novelty = Σ_{s=1}^{k} w_s · u_s / C(k,s)  /  Σ_{s=1}^{k} w_s
      k     = 候选的不同突变单元数
      u_s   = 候选的 s 元单元子集中训练集未出现过的数量
      C(k,s)= 候选 s 元子集总数 (u_s/C(k,s) 即该阶"未见密度")
      → 各阶未见密度的加权和归一, 范围 [0,1], 不同 k 候选可公平比较

    特殊情况:
      - 纯碳源候选 (无突变单元): novelty = 0
      - 复合机制 CLB2[GAL-OE+db] 展开为两个单元 (GAL-OE, db), 同基因
        机制组合同样计入上位性组合
      - k ≤ 12 全子集枚举 (实际上限 k=11: 6 绑定组 + 5 复合机制);
        k > 12 仅评估 1..4 阶 + 完整组合 (防御性兜底, 理论不可达)

    数据隔离: 仅解析 dm.train_dataset_synth 样本名 + self._pool_mass_meta,
    不访问测试集。
    """
    from itertools import combinations

    # ---------- Step 1: 训练集已覆盖的突变单元子集 (bitmask 整数键, 省内存) ----------
    if hasattr(dm, '_mutant_names') and hasattr(dm, '_train_idx_synth'):
        train_names = [str(dm._mutant_names[i]) for i in dm._train_idx_synth]
    else:
        train_names = [f"Train_{i}" for i in range(len(dm.train_dataset_synth))]

    unit_ids = {}   # (gene, mode) -> int id (词表约百余个, 子集压成 bitmask int)
    def _uid(gm):
        if gm not in unit_ids:
            unit_ids[gm] = len(unit_ids)
        return unit_ids[gm]

    MAX_FULL_ENUM_UNITS = 12

    def _sizes_of(k):
        # 评估的尺寸集: k≤12 全枚举 1..k; 超限仅 1..4 阶 + 完整组合
        return range(1, k + 1) if k <= MAX_FULL_ENUM_UNITS else list(range(1, 5)) + [k]

    train_combos = {}   # {size: set(bitmask)}: 训练集出现过的该尺寸全部子集
    for name in train_names:
        desc = _mass_desc_of(name)
        if desc is None:
            continue
        units = sorted({_uid(gm) for gm in _parse_mass_gene_modes(desc)})
        if not units:
            continue
        for s in _sizes_of(len(units)):
            bucket = train_combos.setdefault(s, set())
            for combo in combinations(units, s):
                mask = 0
                for u in combo:
                    mask |= 1 << u
                bucket.add(mask)

    # ---------- Step 2: 候选逐阶加权评分 ----------
    def _size_weight(s):
        # w_1=0.5, w_2=1.0 (核心: 二元上位性), w_3=0.5, 更高阶逐级减半
        return 2.0 ** (-abs(s - 2))

    scores = np.zeros(self._pool_size)
    n_empty = 0
    for i in range(self._pool_size):
        meta = self._pool_mass_meta[i] if i < len(self._pool_mass_meta) else None
        pairs = meta.get('genes', []) if meta else []
        units = sorted({_uid(gm) for gm in pairs})
        if not units:
            n_empty += 1          # 纯碳源样本 → 无组合新颖性
            continue
        w_sum = 0.0
        nov_sum = 0.0
        for s in _sizes_of(len(units)):
            seen_set = train_combos.get(s)
            total = 0
            unseen = 0
            for combo in combinations(units, s):
                mask = 0
                for u in combo:
                    mask |= 1 << u
                total += 1
                if seen_set is None or mask not in seen_set:
                    unseen += 1
            w = _size_weight(s)
            nov_sum += w * (unseen / total)
            w_sum += w
        scores[i] = nov_sum / w_sum

    n_scored = self._pool_size - n_empty
    n_units_seen = len(train_combos.get(1, ()))
    n_unit_pairs = len(train_combos.get(2, ()))
    print(f"[al4pde_novelty] 组合新颖性 v2 (单元级全子集, w_s=2^-|s-2|): 训练集覆盖 "
          f"{n_units_seen} 突变单元 / {n_unit_pairs} 单元对 | scored {n_scored}/{self._pool_size} 候选 | "
          f"range: {scores.min():.4f}~{scores.max():.4f}, mean={scores.mean():.4f}")
    return scores


def _map_family_unc_to_pool(self, family_uncertainty):
    """将训练集家族不确定性映射到候选池样本 (v2.7 多基因版)

    候选的全部突变基因各自查家族不确定性百分位 [0,1], 取均值:
    - 候选 CLB2[KO]+CLN2[GAL-OE] → mean(pct[CLB2], pct[CLN2])
      (样本同时携带两个家族的信息, 补充任一家族均有价值)
    - 训练集中未出现的基因 → 该基因计 0.5 (中性, 由 novelty 主导其排序)
    - 纯碳源候选 → 0 (无家族信号)
    数据隔离: 输入 family_uncertainty 仅来自训练集参数扰动集成。
    """
    fams = sorted(family_uncertainty.keys())
    if len(fams) == 0:
        return np.zeros(self._pool_size)
    vals = np.array([family_uncertainty[f] for f in fams], dtype=np.float64)
    # 家族不确定性 → 百分位排名 [0,1] (消除量纲, 最不确定的家族 → 1.0)
    order = np.argsort(np.argsort(vals))
    ranks = order / max(1, len(fams) - 1)
    fam_pct = dict(zip(fams, ranks))

    scores = np.zeros(self._pool_size)
    for i in range(self._pool_size):
        meta = self._pool_mass_meta[i] if i < len(self._pool_mass_meta) else None
        pairs = meta.get('genes', []) if meta else []
        if pairs:
            # 全部突变基因 (去重) 的家族百分位均值 (v2.7: 不再只看首个基因)
            genes = []
            for gene, _mode in pairs:
                if gene not in genes:
                    genes.append(gene)
            scores[i] = float(np.mean([fam_pct.get(g, 0.5) for g in genes]))
    return scores


def _map_family_priority_train_based(self, family_uncertainty, dm):
    """将训练集家族不确定性映射到候选池样本的 target_priority (完全基于训练集, 零泄露)

    v2.7 多基因版: 候选的全部突变基因各自计算三维信号后取均值
    (候选 CLB2[KO]+CLN2[GAL-OE] 同时按 CLB2 与 CLN2 两个家族评估)。

    target_priority = 0.5*同家族不确定性百分位 + 0.3*候选到训练集同家族参数距离 + 0.2*家族覆盖度
    (三个维度均为: 候选全部突变基因的信号均值; 纯碳源候选按 'CARBON' 伪家族评估)

    数据隔离: 不访问 dm.test_dataset_real, 仅用 dm.train_dataset_synth。
    - 维度1 (0.5): 训练集中同家族样本的不确定性 (模型对该家族不自信 → 优先)
    - 维度2 (0.3): 候选样本到训练集同家族参数中心的距离 (越远 → 训练集覆盖不足 → 优先)
    - 维度3 (0.2): 候选池中同家族样本数 (越少 → 覆盖不足 → 优先)

    Args:
        family_uncertainty: {gene: mean_uncertainty} 来自训练集 (多基因聚合)
        dm: 数据模块 (仅用 train_dataset_synth)

    Returns:
        priorities: (N,) numpy array, 每个候选样本的 target_priority
    """
    priorities = np.zeros(self._pool_size)

    # 1. 不确定性百分位排名 [0, 1]
    unc_values = np.array(list(family_uncertainty.values()))
    unc_percentile = {}
    for name, unc in family_uncertainty.items():
        unc_percentile[name] = float((unc_values <= unc).mean())

    # 2. 家族覆盖度统计 (候选池中每个基因家族的样本数; v2.7: 每个候选
    #    为其全部突变基因各计 1 次, 纯碳源候选计入 'CARBON')
    family_counts = {}
    for name in self._pool_mutant_names:
        genes = _mass_sample_genes(name)
        fams = genes if genes else ['CARBON']
        for fam in fams:
            family_counts[fam] = family_counts.get(fam, 0) + 1
    max_family_count = max(family_counts.values()) if family_counts else 1

    # 3. 训练集参数按基因家族分组 (用于参数距离计算) — 完全来自训练集
    #    v2.7: 每个训练样本的参数归属其全部突变基因 (多基因样本计入多个家族)
    train_P_np = dm.train_dataset_synth.tensors[1].cpu().numpy()
    if hasattr(dm, '_mutant_names') and hasattr(dm, '_train_idx_synth'):
        train_names = [str(dm._mutant_names[i]) for i in dm._train_idx_synth]
    else:
        train_names = [f"Train_{i}" for i in range(len(train_P_np))]

    train_family_params = {}
    for i, name in enumerate(train_names):
        genes = _mass_sample_genes(name)
        fams = genes if genes else ['CARBON']
        for fam in fams:
            train_family_params.setdefault(fam, []).append(train_P_np[i])
    # 每个家族取平均参数向量 (代表该家族的参数中心)
    train_family_center = {f: np.mean(params, axis=0) for f, params in train_family_params.items()}

    # 候选池参数 (归一化空间)
    pool_P_np = self._pool_P.cpu().numpy() if hasattr(self._pool_P, 'cpu') else np.array(self._pool_P)

    for i, name in enumerate(self._pool_mutant_names):
        genes = _mass_sample_genes(name)
        fams = genes if genes else ['CARBON']

        # 维度 1: 全部突变基因的同家族不确定性百分位均值 [0, 1]
        # (基因在训练集中未出现 → 该基因计 0.5 中性值)
        base_priority = float(np.mean([unc_percentile.get(f, 0.5) for f in fams]))

        # 维度 2: 全部突变基因的参数中心距离均值 [0, 1] (越远 → 覆盖不足 → 优先)
        # (基因在训练集中无样本 → 该基因计 0.5 默认值)
        dist_factors = []
        for f in fams:
            if f in train_family_center:
                dist = np.linalg.norm(pool_P_np[i] - train_family_center[f])
                # dist/(1+dist): dist 越大越接近 1, 鼓励探索训练集未覆盖的参数区域
                dist_factors.append(dist / (1.0 + dist))
            else:
                dist_factors.append(0.5)
        distance_factor = float(np.mean(dist_factors))

        # 维度 3: 全部突变基因的家族覆盖度均值 [0, 1] (池中样本越少 → 优先)
        coverage_factor = float(np.mean(
            [1.0 - family_counts.get(f, 1) / max_family_count for f in fams]))

        # 综合 priority: 0.5*不确定性 + 0.3*参数距离 + 0.2*覆盖度
        priorities[i] = 0.5 * base_priority + 0.3 * distance_factor + 0.2 * coverage_factor

    print(f"[MAAL] target_priority 训练集版 (v2.7 多基因聚合): "
          f"0.5*同家族不确定性 + 0.3*参数距离 + 0.2*覆盖度")
    print(f"[MAAL] priority range: {priorities.min():.4f}~{priorities.max():.4f}, "
          f"mean={priorities.mean():.4f}")
    return priorities



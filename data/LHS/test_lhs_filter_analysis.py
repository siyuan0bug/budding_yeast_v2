#!/usr/bin/env python3
"""
LHS 采样过滤阶段损失分析工具

基于 lhs_v1_origin.py，追踪每个过滤操作对各家族 LHS 样本的影响。
默认条件：采样参数不做任何额外处理，直接走原始管线。

追踪的过滤阶段:
  Stage 1: 等价碰撞拦截 + 纯离散去重        (LHS 生成阶段)
  Stage 2: 超时 + Discarded_Mass           (ODE 模拟阶段)
  Stage 3: 动态浓度剔除 (2× Real Mutant)   (模拟后处理阶段)

用法:
  python test_lhs_filter_analysis.py                                # 默认配置 (200样本)
  python test_lhs_filter_analysis.py --remove ESP1T                 # 移除 ESP1T 离散约束
  python test_lhs_filter_analysis.py --remove init_CDH1T init_CDH1 # 移除 CDH1 对绑定
  python test_lhs_filter_analysis.py --num-samples 1000             # 完整采样
  python test_lhs_filter_analysis.py --skip-simulation              # 只分析 LHS 生成阶段
"""

import numpy as np
import time
import os
import sys
import argparse
from collections import defaultdict
import concurrent.futures
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhs_v1_origin as origin
from lhs_v1_origin import (
    get_default_params, simulate_mutant, mutant_rules,
    build_discrete_pools, SIM_TIMEOUT
)

DEFAULT_DISCRETE = [
    'init_CDH1T', 'init_CDH1',
    'mdt', 'BUB2_active', 'init_BUB2', 'is_mutant_104', 'ESP1T'
]


# ==========================================
# Stage 1: 带统计的 LHS 生成
# ==========================================
def generate_lhs_with_stats(num_lhs_samples, discrete_params):
    """
    复制原始 generate_lhs_tasks 逻辑，添加每阶段统计。
    返回 (lhs_tasks_dict, stats)
    stats[family] = {target, candidates, equiv_filtered, dedup_filtered, valid, continuous, discrete}
    """
    origin.DISCRETE_STRICT_PARAMS = list(discrete_params)

    single_pools, cdh1_pairs, bub2_pairs = build_discrete_pools(mutant_rules)

    defaults = get_default_params()
    defaults.update({
        'init_CDH1T': 1.0, 'init_CDH1': 0.9304992,
        'init_BUB2': 0.2, 'BUB2_active': 1.0, 'is_mutant_104': False
    })

    all_historical_values = {}
    for rules in mutant_rules.values():
        for p_name, p_val in rules.items():
            if p_name not in all_historical_values:
                all_historical_values[p_name] = set()
            all_historical_values[p_name].add(p_val)

    lhs_tasks_dict = {}
    stats = defaultdict(lambda: {
        'target': 0, 'candidates': 0,
        'equiv_filtered': 0, 'dedup_filtered': 0,
        'valid': 0, 'continuous': 0, 'discrete': 0
    })

    pure_wt = ['1_WT_Glc', '2.1_WT_Gal', '2.2_WT_Raff']

    for m_name, original_rules in mutant_rules.items():
        if m_name in pure_wt:
            continue

        mutated_params = list(original_rules.keys())
        if not mutated_params:
            continue

        strict_mutated = [p for p in mutated_params if p in discrete_params]
        continuous_mutated = [p for p in mutated_params if p not in discrete_params]
        has_bub2 = any(p in ['BUB2_active', 'init_BUB2'] for p in strict_mutated)

        stats[m_name]['continuous'] = len(continuous_mutated)
        stats[m_name]['discrete'] = len(strict_mutated)

        # 智能截断
        if not continuous_mutated and not has_bub2:
            max_comb = 1
            if 'init_CDH1T' in strict_mutated or 'init_CDH1' in strict_mutated:
                max_comb *= len(cdh1_pairs)
            for p in strict_mutated:
                if p not in ['init_CDH1T', 'init_CDH1', 'init_BUB2', 'BUB2_active'] and p in single_pools:
                    max_comb *= len(single_pools[p])
            actual_samples = min(num_lhs_samples, max(1, max_comb - 1))
        else:
            actual_samples = num_lhs_samples

        stats[m_name]['target'] = actual_samples
        if actual_samples <= 0:
            continue

        candidate_count = actual_samples * 10

        # 连续参数 LHS
        continuous_samples = {}
        if continuous_mutated:
            sampler = qmc.LatinHypercube(d=len(continuous_mutated), seed=int(time.time() * 1000) % 2**32)
            lhs_matrix = sampler.random(n=candidate_count)
            for i, p_name in enumerate(continuous_mutated):
                hist_vals = sorted(list(all_historical_values.get(p_name, set())))
                def_val = defaults.get(p_name, 0.0)
                all_known = hist_vals + [def_val]
                min_v, max_v = min(all_known), max(all_known)
                l_b = 0.0 if min_v <= 0 else min_v * 0.1
                u_b = 0.01 if max_v <= 0 else max_v * 1.5
                if u_b <= l_b:
                    u_b = l_b + 0.01
                continuous_samples[p_name] = l_b + lhs_matrix[:, i] * (u_b - l_b)

        # BUB2 同频缩放
        if has_bub2:
            bub2_coeffs = qmc.LatinHypercube(d=1).random(n=candidate_count).flatten() * 1.5
            np.random.shuffle(bub2_coeffs)

        valid_count = 0
        seen = set()

        for k in range(candidate_count):
            stats[m_name]['candidates'] += 1

            sample = {}
            for p_name in continuous_mutated:
                sample[p_name] = continuous_samples[p_name][k]

            sampled_cdh1, sampled_bub2 = False, False
            for dp in strict_mutated:
                if dp in ['init_CDH1T', 'init_CDH1']:
                    if not sampled_cdh1:
                        pair = cdh1_pairs[np.random.choice(len(cdh1_pairs))]
                        sample['init_CDH1T'] = pair[0]
                        sample['init_CDH1'] = pair[1]
                        sampled_cdh1 = True
                elif dp in ['init_BUB2', 'BUB2_active']:
                    if not sampled_bub2:
                        pair = bub2_pairs[np.random.choice(len(bub2_pairs))]
                        coeff = bub2_coeffs[k]
                        sample['init_BUB2'] = pair[0] * coeff
                        sample['BUB2_active'] = pair[1] * coeff
                        sampled_bub2 = True
                else:
                    sample[dp] = np.random.choice(single_pools[dp])

            for p, val in original_rules.items():
                if p not in sample:
                    sample[p] = val

            # 等价碰撞拦截
            is_identical = True
            for p_name in mutated_params:
                if not np.isclose(sample[p_name], original_rules[p_name], rtol=1e-5, atol=1e-8):
                    is_identical = False
                    break
            if is_identical:
                stats[m_name]['equiv_filtered'] += 1
                continue

            # 纯离散去重
            if not continuous_mutated and not has_bub2:
                config_tuple = tuple(sample.get(p) for p in sorted(mutated_params))
                if config_tuple in seen:
                    stats[m_name]['dedup_filtered'] += 1
                    continue
                seen.add(config_tuple)

            lhs_tasks_dict[f"{m_name}_LHS_{valid_count:03d}"] = sample
            valid_count += 1
            stats[m_name]['valid'] += 1

            if valid_count >= actual_samples:
                break

    return lhs_tasks_dict, stats


# ==========================================
# Stage 2: 带统计的并行 ODE 模拟
# ==========================================
def simulate_with_stats(all_tasks, tf, num_steps):
    """
    并行模拟，追踪超时和 mass 丢弃。
    只存储 max values (39,) 用于浓度剔除，不存完整时序以节省内存。
    """
    def get_family(name):
        if '_LHS_' in name:
            return name.split('_LHS_')[0]
        return name

    sim_stats = defaultdict(lambda: {
        'lhs_input': 0, 'timeout': 0, 'mass_discarded': 0,
        'fail': 0, 'success': 0
    })
    results_max = {}

    ordered_names = list(all_tasks.keys())

    for name in ordered_names:
        if '_LHS_' in name:
            sim_stats[get_family(name)]['lhs_input'] += 1

    total_cores = os.cpu_count()
    safe_cores = max(1, total_cores // 4)

    print(f"  CPU 核心: {safe_cores}/{total_cores}, 任务数: {len(ordered_names)}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_cores) as executor:
        future_to_name = {
            executor.submit(simulate_mutant, name, all_tasks[name], tf, num_steps): name
            for name in ordered_names
        }

        for idx, future in enumerate(concurrent.futures.as_completed(future_to_name)):
            name = future_to_name[future]
            family = get_family(name)
            is_lhs = '_LHS_' in name

            try:
                _, y_uniform, bio_label, _ = future.result(timeout=SIM_TIMEOUT + 10.0)

                if y_uniform is not None:
                    results_max[name] = np.max(y_uniform, axis=1)
                    if is_lhs:
                        sim_stats[family]['success'] += 1
                elif bio_label == "Discarded_Mass":
                    if is_lhs:
                        sim_stats[family]['mass_discarded'] += 1
                else:
                    if is_lhs:
                        sim_stats[family]['fail'] += 1

            except concurrent.futures.TimeoutError:
                if is_lhs:
                    sim_stats[family]['timeout'] += 1
            except Exception:
                if is_lhs:
                    sim_stats[family]['fail'] += 1

            if (idx + 1) % 2000 == 0:
                print(f"  进度: {idx + 1} / {len(ordered_names)}")

    return results_max, sim_stats


# ==========================================
# Stage 3: 动态浓度剔除
# ==========================================
def cull_with_stats(results_max, ordered_names):
    """
    动态浓度剔除：LHS 样本任何变量 > 2× Real Mutant 最大值即销毁。
    返回 cull_stats[family] = {culled, survived}
    """
    def get_family(name):
        if '_LHS_' in name:
            return name.split('_LHS_')[0]
        return name

    real_max_vals = []
    for name in ordered_names:
        if name in results_max and '_LHS_' not in name:
            real_max_vals.append(results_max[name])

    if not real_max_vals:
        print("  ⚠️ 无 Real Mutant 基准，跳过浓度剔除")
        return defaultdict(lambda: {'culled': 0, 'survived': 0}), np.zeros(39), np.zeros(39)

    real_envelope = np.max(real_max_vals, axis=0)
    thresholds = np.maximum(real_envelope * 2.0, 2.0)

    cull_stats = defaultdict(lambda: {'culled': 0, 'survived': 0})

    for name in ordered_names:
        if name not in results_max or '_LHS_' not in name:
            continue

        family = get_family(name)
        lhs_max = results_max[name]

        is_culled = False
        for var_idx in range(1, 39):
            if lhs_max[var_idx] > thresholds[var_idx]:
                is_culled = True
                break

        if is_culled:
            cull_stats[family]['culled'] += 1
        else:
            cull_stats[family]['survived'] += 1

    return cull_stats, real_envelope, thresholds


# ==========================================
# Report
# ==========================================
def print_stage1(stats):
    print("\n" + "=" * 110)
    print("Stage 1: LHS 生成阶段 (等价碰撞拦截 + 纯离散去重)")
    print("=" * 110)
    hdr = f"{'Family':<42}|{'Cont':>4} {'Disc':>4}|{'Target':>7} {'Cand':>7} {'Equiv':>6} {'Dedup':>6} {'Valid':>6} {'Loss%':>6}"
    print(hdr)
    print("-" * 110)

    t = defaultdict(int)
    for m in sorted(stats.keys()):
        s = stats[m]
        loss = 100 * (s['equiv_filtered'] + s['dedup_filtered']) / s['candidates'] if s['candidates'] > 0 else 0
        print(f"{m:<42}|{s['continuous']:>4} {s['discrete']:>4}|{s['target']:>7} {s['candidates']:>7} "
              f"{s['equiv_filtered']:>6} {s['dedup_filtered']:>6} {s['valid']:>6} {loss:>5.1f}%")
        for k in ['target', 'candidates', 'equiv_filtered', 'dedup_filtered', 'valid']:
            t[k] += s[k]

    print("-" * 110)
    total_loss = 100 * (t['equiv_filtered'] + t['dedup_filtered']) / t['candidates'] if t['candidates'] > 0 else 0
    print(f"{'TOTAL':<42}|{'':>4} {'':>4}|{t['target']:>7} {t['candidates']:>7} "
          f"{t['equiv_filtered']:>6} {t['dedup_filtered']:>6} {t['valid']:>6} {total_loss:>5.1f}%")
    print("=" * 110)

    # 高亮损失率高的家族
    high_loss = [(m, s) for m, s in stats.items()
                 if s['candidates'] > 0 and
                 100 * (s['equiv_filtered'] + s['dedup_filtered']) / s['candidates'] > 5]
    if high_loss:
        print("\n  ⚠️ 损失率 > 5% 的家族:")
        for m, s in sorted(high_loss, key=lambda x: -x[1]['equiv_filtered'] - x[1]['dedup_filtered']):
            loss = 100 * (s['equiv_filtered'] + s['dedup_filtered']) / s['candidates']
            print(f"    {m}: equiv={s['equiv_filtered']}, dedup={s['dedup_filtered']}, loss={loss:.1f}%")

    # 未达目标的家族
    unmet = [(m, s) for m, s in stats.items() if s['valid'] < s['target']]
    if unmet:
        print(f"\n  ⚠️ 未达目标的家族 ({len(unmet)}个):")
        for m, s in sorted(unmet, key=lambda x: x[1]['target'] - x[1]['valid'], reverse=True):
            gap = s['target'] - s['valid']
            print(f"    {m}: target={s['target']}, valid={s['valid']}, gap={gap}")


def print_stage2(sim_stats, stage1_stats):
    print("\n" + "=" * 95)
    print("Stage 2: ODE 模拟阶段 (超时 + Discarded_Mass)")
    print("=" * 95)
    print(f"{'Family':<42}|{'Input':>7} {'Timeout':>7} {'Mass':>6} {'Fail':>6} {'Succ':>6} {'Succ%':>6}")
    print("-" * 95)

    t = defaultdict(int)
    for m in sorted(sim_stats.keys()):
        s = sim_stats[m]
        if m not in stage1_stats or stage1_stats[m]['valid'] == 0:
            continue
        succ_rate = 100 * s['success'] / s['lhs_input'] if s['lhs_input'] > 0 else 0
        print(f"{m:<42}|{s['lhs_input']:>7} {s['timeout']:>7} {s['mass_discarded']:>6} "
              f"{s['fail']:>6} {s['success']:>6} {succ_rate:>5.1f}%")
        for k in ['lhs_input', 'timeout', 'mass_discarded', 'fail', 'success']:
            t[k] += s[k]

    print("-" * 95)
    succ_rate = 100 * t['success'] / t['lhs_input'] if t['lhs_input'] > 0 else 0
    print(f"{'TOTAL':<42}|{t['lhs_input']:>7} {t['timeout']:>7} {t['mass_discarded']:>6} "
          f"{t['fail']:>6} {t['success']:>6} {succ_rate:>5.1f}%")
    print("=" * 95)


def print_stage3(cull_stats, sim_stats, real_envelope, thresholds):
    print("\n" + "=" * 85)
    print("Stage 3: 动态浓度剔除 (LHS > 2× Real Mutant 最大值)")
    print("=" * 85)

    # 打印阈值
    var_names = ['MASS', 'CLN2', 'CLB2', 'CLB5', 'SIC1', 'CDC6', 'C2', 'C5',
                 'F2', 'F5', 'SIC1P', 'C2P', 'C5P', 'CDC6P', 'F2P', 'F5P',
                 'SWI5T', 'SWI5', 'IEP', 'CDC20T', 'CDC20', 'CDH1T', 'CDH1',
                 'CDC14T', 'CDC14', 'NET1T', 'NET1', 'RENT', 'TEM1', 'CDC15',
                 'PPX', 'PDS1', 'ESP1', 'ORI', 'BUD', 'SPN', 'Vi20', 'lte1', 'BUB2']
    print(f"  Real Mutant 包络 (top-5): ", end="")
    top5 = np.argsort(real_envelope[1:])[-5:][::-1] + 1
    print(", ".join(f"{var_names[i]}={real_envelope[i]:.3f}" for i in top5))
    print(f"  动态阈值 (top-5):         ", end="")
    print(", ".join(f"{var_names[i]}={thresholds[i]:.3f}" for i in top5))

    print(f"\n{'Family':<42}|{'Succ':>6} {'Culled':>7} {'Surv':>6} {'Cull%':>6}")
    print("-" * 85)

    t_culled = t_survived = 0
    for m in sorted(cull_stats.keys()):
        s = cull_stats[m]
        if m not in sim_stats or sim_stats[m]['success'] == 0:
            continue
        cull_rate = 100 * s['culled'] / (s['culled'] + s['survived']) if (s['culled'] + s['survived']) > 0 else 0
        print(f"{m:<42}|{sim_stats[m]['success']:>6} {s['culled']:>7} {s['survived']:>6} {cull_rate:>5.1f}%")
        t_culled += s['culled']
        t_survived += s['survived']

    total = t_culled + t_survived
    cull_rate = 100 * t_culled / total if total > 0 else 0
    print("-" * 85)
    print(f"{'TOTAL':<42}|{total:>6} {t_culled:>7} {t_survived:>6} {cull_rate:>5.1f}%")
    print("=" * 85)


def print_summary(stage1_stats, sim_stats, cull_stats):
    print("\n" + "=" * 80)
    print("Pipeline 完整损失链路 (LHS 样本)")
    print("=" * 80)

    t_cand = sum(s['candidates'] for s in stage1_stats.values())
    t_equiv = sum(s['equiv_filtered'] for s in stage1_stats.values())
    t_dedup = sum(s['dedup_filtered'] for s in stage1_stats.values())
    t_valid = sum(s['valid'] for s in stage1_stats.values())
    t_target = sum(s['target'] for s in stage1_stats.values())

    lhs_in = sum(s['lhs_input'] for s in sim_stats.values())
    lhs_to = sum(s['timeout'] for s in sim_stats.values())
    lhs_mass = sum(s['mass_discarded'] for s in sim_stats.values())
    lhs_fail = sum(s['fail'] for s in sim_stats.values())
    lhs_succ = sum(s['success'] for s in sim_stats.values())

    t_culled = sum(s['culled'] for s in cull_stats.values())
    t_survived = sum(s['survived'] for s in cull_stats.values())

    stages = [
        ("目标采样数",       t_target, t_target,    "基准"),
        ("候选生成",         t_cand,   t_cand,      "candidate_count=target×10"),
        (" ├ 等价碰撞拦截",  t_equiv,  t_cand,      "样本与母体完全一致"),
        (" ├ 纯离散去重",    t_dedup,  t_cand,      "纯离散重复配置"),
        (" └ 有效LHS任务",   t_valid,  t_target,    "进入ODE模拟"),
        ("ODE 超时",         lhs_to,   lhs_in,      f"超过{SIM_TIMEOUT}s"),
        ("ODE Mass丢弃",     lhs_mass, lhs_in,      "max_mass>10"),
        ("ODE 其他失败",     lhs_fail, lhs_in,      "异常"),
        ("ODE 成功",         lhs_succ, lhs_in,      "进入浓度检查"),
        ("浓度剔除",         t_culled, lhs_succ,    ">2×RealMutant"),
        ("最终数据集",       t_survived, lhs_succ,  "保存到npz"),
    ]

    print(f"{'Stage':<22} | {'Count':>8} | {'Rate':>8} | {'Base':>8} | {'说明':<24}")
    print("-" * 80)
    for name, count, base, desc in stages:
        if base > 0:
            rate = f"{count/base*100:.2f}%"
        else:
            rate = "N/A"
        print(f"{name:<22} | {count:>8} | {rate:>8} | {base:>8} | {desc:<24}")
    print("=" * 80)

    # 总有效率
    if t_target > 0:
        final_rate = t_survived / t_target * 100
        print(f"\n  最终有效率: {t_survived}/{t_target} = {final_rate:.2f}%")
        print(f"  总损失:     {t_target - t_survived}/{t_target} = {100-final_rate:.2f}%")


# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LHS 采样过滤阶段损失分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_lhs_filter_analysis.py                                # 默认 (200样本/突变体)
  python test_lhs_filter_analysis.py --remove ESP1T                 # ESP1T 变为连续参数
  python test_lhs_filter_analysis.py --remove init_CDH1T init_CDH1   # CDH1 对绑定解除
  python test_lhs_filter_analysis.py --num-samples 1000             # 完整采样
  python test_lhs_filter_analysis.py --skip-simulation              # 只看LHS生成阶段
        """)
    parser.add_argument('--remove', nargs='*', default=[],
                        help='从离散名单移除的参数 (如: ESP1T  或  init_CDH1T init_CDH1)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='每个突变体的 LHS 采样数 (默认 200，原始=1000)')
    parser.add_argument('--skip-simulation', action='store_true',
                        help='跳过 ODE 模拟，只分析 LHS 生成阶段')
    parser.add_argument('--tf', type=int, default=210, help='模拟总时长 min (默认 210)')
    parser.add_argument('--steps', type=int, default=500, help='采样步数 (默认 500)')
    args = parser.parse_args()

    # 配置离散参数
    discrete_params = list(DEFAULT_DISCRETE)
    for p in args.remove:
        if p in discrete_params:
            discrete_params.remove(p)
            print(f"  [配置] 已从离散名单移除: {p} (→ 变为连续参数)")
        else:
            print(f"  [配置] 参数 {p} 不在离散名单中，跳过")

    print(f"\n{'='*60}")
    print(f"LHS 采样过滤损失分析")
    print(f"{'='*60}")
    print(f"  离散参数 ({len(discrete_params)}): {discrete_params}")
    print(f"  LHS 采样数/突变体: {args.num_samples}")
    print(f"  模拟参数: tf={args.tf}min, steps={args.steps}")
    print(f"  超时阈值: {SIM_TIMEOUT}s")

    # Stage 1
    print(f"\n{'='*60}")
    print("Stage 1: LHS 生成中...")
    print(f"{'='*60}")
    start = time.time()
    tasks, stage1_stats = generate_lhs_with_stats(args.num_samples, discrete_params)
    print(f"  完成: {len(tasks)} 个 LHS 任务, 耗时 {time.time()-start:.1f}s")

    print_stage1(stage1_stats)

    if args.skip_simulation:
        print("\n  --skip-simulation 已指定，跳过 ODE 模拟")
        print_summary(stage1_stats,
                      defaultdict(lambda: {'lhs_input': 0, 'timeout': 0, 'mass_discarded': 0, 'fail': 0, 'success': 0}),
                      defaultdict(lambda: {'culled': 0, 'survived': 0}))
        sys.exit(0)

    # 合并 real + LHS
    all_tasks = {}
    all_tasks.update(mutant_rules)
    all_tasks.update(tasks)

    # Stage 2
    print(f"\n{'='*60}")
    print("Stage 2: ODE 模拟中...")
    print(f"{'='*60}")
    start = time.time()
    results_max, sim_stats = simulate_with_stats(all_tasks, args.tf, args.steps)
    print(f"  完成: 耗时 {time.time()-start:.1f}s")

    print_stage2(sim_stats, stage1_stats)

    # Stage 3
    print(f"\n{'='*60}")
    print("Stage 3: 动态浓度剔除中...")
    print(f"{'='*60}")
    cull_stats, real_envelope, thresholds = cull_with_stats(results_max, list(all_tasks.keys()))

    print_stage3(cull_stats, sim_stats, real_envelope, thresholds)

    # Summary
    print_summary(stage1_stats, sim_stats, cull_stats)

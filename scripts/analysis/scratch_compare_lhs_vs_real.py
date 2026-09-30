#!/usr/bin/env python3
import re
import sys
from collections import Counter
import numpy as np
import pandas as pd

sys.path.append('budding_yeast_v2/data')
import mutant_aware as ma

p_lhs = 'budding_yeast_v2/experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/triple_dynamic_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_lhs_test_metrics.csv'
p_real = 'budding_yeast_v2/experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/triple_dynamic_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_mutant_metrics.csv'

df_lhs = pd.read_csv(p_lhs)
df_real = pd.read_csv(p_real)

def parse_lhs_mutant(mut_str):
    m = re.search(r'RANDOM_(.+)_L(\d+)_(\d+)', mut_str)
    if not m:
        return [], 0
    body = m.group(1)
    level = int(m.group(2))
    tokens = body.split('+')
    parsed = []
    for tok in tokens:
        sub = re.match(r'([A-Za-z0-9_\-]+)\[([A-Za-z0-9_\-\+]+)\]', tok)
        if sub:
            g, mode = sub.group(1), sub.group(2)
            if g != 'MDT':
                parsed.append((g, mode))
    return parsed, level

def parse_real_mutant(m_str):
    mid = m_str.split('_')[0]
    if mid not in ma.REAL_MUTANTS:
        return [], 0
    name, ovr = ma.REAL_MUTANTS[mid]
    muts = []
    if 'C0' in ovr:
        if ovr['C0'] == 0.0: muts.append(('CLN3', 'KO'))
        elif ovr['C0'] > 1.0: muts.append(('CLN3', 'GAL-OE'))
    if 'B0' in ovr:
        if ovr['B0'] == 0.0: muts.append(('BCK2', 'KO'))
        elif ovr['B0'] > 0.1: muts.append(('BCK2', 'MC'))
    if 'ksn2pp' in ovr or 'ksn2p' in ovr:
        if ovr.get('ksn2pp') == 0.0: muts.append(('CLN2', 'KO'))
        if ovr.get('ksn2p', 0) > 0.05: muts.append(('CLN2', 'GAL-OE'))
        if ovr.get('ksn2pp', 0) > 0.2: muts.append(('CLN2', 'MC'))
    if 'ksc1p' in ovr or 'ksc1pp' in ovr or 'kd3c1' in ovr:
        if ovr.get('ksc1pp') == 0.0 and ovr.get('ksc1p') == 0.0: muts.append(('SIC1', 'KO'))
        if ovr.get('ksc1p', 0) > 0.05: muts.append(('SIC1', 'GAL-OE'))
        if ovr.get('kd3c1') == 0.0: muts.append(('SIC1', 'db'))
    if 'kscdh' in ovr or 'init_CDH1T' in ovr or 'kicdhpp' in ovr:
        if ovr.get('kscdh') == 0.0 and ovr.get('init_CDH1T') == 0.0: muts.append(('CDH1', 'KO'))
        if ovr.get('kicdhpp') == 0.0: muts.append(('CDH1', 'CA'))
    if 'ksb5p' in ovr or 'ksb5pp' in ovr:
        if ovr.get('ksb5p') == 0.0 and ovr.get('ksb5pp') == 0.0: muts.append(('CLB5', 'KO'))
        elif ovr.get('ksb5p', 0) > 0.005 or ovr.get('ksb5pp', 0) > 0.01:
            muts.append(('CLB5', 'GAL-OE' if 'GAL' in name else 'MC'))
    if 'ksb2p' in ovr or 'ksb2pp' in ovr or 'kdb2P' in ovr or 'kdb2pp' in ovr:
        if ovr.get('ksb2p') == 0.0 and ovr.get('ksb2pp') == 0.0: muts.append(('CLB2', 'KO'))
        if ovr.get('ksb2p', 0) > 0.01: muts.append(('CLB2', 'GAL-OE'))
        if ovr.get('kdb2P') == 0.0: muts.append(('CLB2', 'db'))
    if 'ks20p' in ovr or 'ks20pp' in ovr:
        if ovr.get('ks20pp') == 0.0 and ovr.get('ks20p') == 0.0: muts.append(('CDC20', 'KO'))
        elif ovr.get('ks20p', 0) > 1.0: muts.append(('CDC20', 'GAL-OE'))
    if 'ksf6p' in ovr or 'ksf6pp' in ovr:
        if ovr.get('ksf6p') == 0.0 and ovr.get('ksf6pp') == 0.0: muts.append(('CDC6', 'KO'))
        elif ovr.get('ksf6p', 0) > 0.05: muts.append(('CDC6', 'MC' if 'multi' in name else 'GAL-OE'))
    if 'ks1pdspp' in ovr or 'ks2pdspp' in ovr:
        if ovr.get('ks1pdspp') == 0.0: muts.append(('PDS1', 'KO'))
        if 'GAL-PDS1' in name or 'PDS1-db' in name: muts.append(('PDS1', 'GAL-OE' if 'GAL' in name else 'db'))
    if 'CDC14' in name or 'cdc14' in name:
        if 'ts' in name: muts.append(('CDC14', 'ts'))
        if 'GAL' in name: muts.append(('CDC14', 'GAL-OE'))
    if 'CDC15' in name or 'cdc15' in name:
        if 'ts' in name: muts.append(('CDC15', 'ts'))
        if 'GAL' in name: muts.append(('CDC15', 'GAL-OE'))
    if 'NET1' in name or 'net1' in name:
        if 'GAL' in name: muts.append(('NET1', 'GAL-OE'))
        if 'TAB6' in name: muts.append(('NET1', 'TAB6-1'))
        if 'ts' in name: muts.append(('NET1', 'ts'))
    if 'swi5' in name: muts.append(('SWI5', 'KO'))
    if 'mad2' in name: muts.append(('MAD2', 'KO'))
    if 'bub2' in name: muts.append(('BUB2', 'KO'))
    if 'esp1' in name: muts.append(('ESP1', 'ts'))
    if 'tem1' in name: muts.append(('TEM1', 'ts' if 'ts' in name else 'KO'))
    
    seen = set()
    uniq = []
    for g, m in muts:
        if (g, m) not in seen:
            seen.add((g, m))
            uniq.append((g, m))
    return uniq, len(uniq)

# Parse all
lhs_list = []
for idx, r in df_lhs.iterrows():
    m, lvl = parse_lhs_mutant(r['Mutant'])
    lhs_list.append({
        'Mutant': r['Mutant'], 'Pattern': r['Pattern'], 'Relative_L2': r['Relative_L2'],
        'MAE': r['MAE'], 'muts': m, 'level': lvl,
        'genes': [g for g, mod in m], 'modes': [mod for g, mod in m]
    })
df_l = pd.DataFrame(lhs_list)

real_list = []
for idx, r in df_real.iterrows():
    m, lvl = parse_real_mutant(r['Mutant'])
    real_list.append({
        'Mutant': r['Mutant'], 'Pattern': r['Pattern'], 'Relative_L2': r['Relative_L2'],
        'MAE': r['MAE'], 'muts': m, 'level': lvl,
        'genes': [g for g, mod in m], 'modes': [mod for g, mod in m]
    })
df_r = pd.DataFrame(real_list)

# Define Badcases
# Real: Top 15 worst (Rel L2 >= 0.179) and Rel L2 > 0.3 (6)
real_bad = df_r.sort_values(by='Relative_L2', ascending=False).head(15)
real_good = df_r.sort_values(by='Relative_L2', ascending=True).head(30)

# LHS: Top 2% (Top 200, Rel L2 > 0.33) and Top 500 (Rel L2 > 0.20)
lhs_top200 = df_l.sort_values(by='Relative_L2', ascending=False).head(200)
lhs_top500 = df_l.sort_values(by='Relative_L2', ascending=False).head(500)
lhs_good = df_l.sort_values(by='Relative_L2', ascending=True).head(500)

print('=' * 80)
print('1. MUTATION ORDER / LEVEL COMPARISON (阶数对比)')
print('=' * 80)
print('Real Badcases (Top 15):', Counter(real_bad['level']))
print('Real All (126):', Counter(df_r['level']))
print('LHS Badcases (Top 200):', Counter(lhs_top200['level']))
print('LHS All (10,001):', Counter(df_l['level']))

print('\n' + '=' * 80)
print('2. DYNAMICAL PATTERN COMPARISON (表型分布对比)')
print('=' * 80)
print('Real Badcases (Top 15):')
for k, v in Counter(real_bad['Pattern']).items():
    print(f'  {k}: {v} ({v/len(real_bad)*100:.1f}%)')
print('Real All (126):')
for k, v in Counter(df_r['Pattern']).items():
    print(f'  {k}: {v} ({v/len(df_r)*100:.1f}%)')

print('LHS Badcases (Top 200):')
for k, v in Counter(lhs_top200['Pattern']).items():
    print(f'  {k}: {v} ({v/len(lhs_top200)*100:.1f}%)')
print('LHS All (10,001):')
for k, v in Counter(df_l['Pattern']).items():
    print(f'  {k}: {v} ({v/len(df_l)*100:.1f}%)')

print('\n' + '=' * 80)
print('3. GENE FREQUENCY COMPARISON (受累基因频率对比)')
print('=' * 80)
# Count gene appearances
real_bad_genes = Counter([g for row in real_bad['genes'] for g in row])
real_all_genes = Counter([g for row in df_r['genes'] for g in row])

lhs_bad_genes = Counter([g for row in lhs_top200['genes'] for g in row])
lhs_all_genes = Counter([g for row in df_l['genes'] for g in row])

# Calculate enrichment and percentage
all_gene_keys = sorted(list(set(list(lhs_bad_genes.keys()) + list(real_bad_genes.keys()))))
comp_rows = []
for g in all_gene_keys:
    r_bad_cnt = real_bad_genes.get(g, 0)
    r_bad_pct = r_bad_cnt / len(real_bad) * 100
    r_all_pct = real_all_genes.get(g, 0) / len(df_r) * 100
    
    l_bad_cnt = lhs_bad_genes.get(g, 0)
    l_bad_pct = l_bad_cnt / len(lhs_top200) * 100
    l_all_pct = lhs_all_genes.get(g, 0) / len(df_l) * 100
    
    # Enrichment ratio in LHS (bad / all)
    l_enrich = l_bad_pct / (l_all_pct + 1e-6)
    r_enrich = r_bad_pct / (r_all_pct + 1e-6)
    
    comp_rows.append({
        'Gene': g,
        'LHS_Bad_Count': l_bad_cnt,
        'LHS_Bad_Pct': l_bad_pct,
        'LHS_All_Pct': l_all_pct,
        'LHS_Enrich': l_enrich,
        'Real_Bad_Count': r_bad_cnt,
        'Real_Bad_Pct': r_bad_pct,
        'Real_All_Pct': r_all_pct,
        'Real_Enrich': r_enrich
    })

df_comp = pd.DataFrame(comp_rows)
df_comp_sorted = df_comp.sort_values(by='LHS_Bad_Pct', ascending=False)
print(f"{'Gene':<12} | {'LHS Bad %':<10} | {'LHS Base %':<10} | {'LHS Enrich':<10} | {'Real Bad %':<10} | {'Real Base %':<10} | {'Real Enrich':<10}")
print('-' * 85)
for idx, r in df_comp_sorted.iterrows():
    print(f"{r['Gene']:<12} | {r['LHS_Bad_Pct']:>9.1f}% | {r['LHS_All_Pct']:>9.1f}% | {r['LHS_Enrich']:>10.2f}x | {r['Real_Bad_Pct']:>9.1f}% | {r['Real_All_Pct']:>9.1f}% | {r['Real_Enrich']:>10.2f}x")

print('\n' + '=' * 80)
print('4. MUTATION TYPE COMPARISON (突变类型分布对比)')
print('=' * 80)
def generalize_mode(m):
    if m in ['KO']: return 'KO (Deletion)'
    if m in ['GAL-OE', 'MC']: return 'OE/MC (Overexpression)'
    if m in ['ts', 'DAmP', 'pLOF']: return 'Partial LOF/ts'
    if m in ['db', 'CA', 'PM', 'TAB6-1', 'scc1-RR']: return 'Constitutive/Structural'
    return 'Other'

lhs_modes = Counter([generalize_mode(m) for row in lhs_top200['modes'] for m in row])
real_modes = Counter([generalize_mode(m) for row in real_bad['modes'] for m in row])
lhs_all_modes = Counter([generalize_mode(m) for row in df_l['modes'] for m in row])
real_all_modes = Counter([generalize_mode(m) for row in df_r['modes'] for m in row])

for m in ['KO (Deletion)', 'OE/MC (Overexpression)', 'Partial LOF/ts', 'Constitutive/Structural']:
    l_b = lhs_modes.get(m, 0) / sum(lhs_modes.values()) * 100
    l_a = lhs_all_modes.get(m, 0) / sum(lhs_all_modes.values()) * 100
    r_b = real_modes.get(m, 0) / sum(real_modes.values()) * 100
    r_a = real_all_modes.get(m, 0) / sum(real_all_modes.values()) * 100
    print(f"{m:<25} | LHS Bad: {l_b:5.1f}% (Base: {l_a:5.1f}%) | Real Bad: {r_b:5.1f}% (Base: {r_a:5.1f}%)")

print('\n' + '=' * 80)
print('5. TOP CO-OCCURRING GENE PAIRS IN LHS BADCASES VS REAL BADCASES')
print('=' * 80)
lhs_pairs = Counter()
for genes in lhs_top200['genes']:
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            pair = tuple(sorted([genes[i], genes[j]]))
            lhs_pairs[pair] += 1

real_pairs = Counter()
for genes in real_bad['genes']:
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            pair = tuple(sorted([genes[i], genes[j]]))
            real_pairs[pair] += 1

print('Top 15 Gene Pairs in LHS Badcases:')
for p, cnt in lhs_pairs.most_common(15):
    print(f"  {p[0]:<10} + {p[1]:<10} : {cnt:3d} ({cnt/len(lhs_top200)*100:.1f}%)")

print('\nTop 15 Gene Pairs in Real Badcases:')
for p, cnt in real_pairs.most_common(15):
    print(f"  {p[0]:<10} + {p[1]:<10} : {cnt:3d} ({cnt/len(real_bad)*100:.1f}%)")

# Correlation between LHS gene frequency and Real gene frequency
overlap_genes = [g for g in ma.GENE_NAMES if g in df_comp['Gene'].values]
lhs_pcts = [df_comp[df_comp['Gene'] == g]['LHS_Bad_Pct'].values[0] for g in overlap_genes]
real_pcts = [df_comp[df_comp['Gene'] == g]['Real_Bad_Pct'].values[0] for g in overlap_genes]
corr = np.corrcoef(lhs_pcts, real_pcts)[0, 1]
print('\n' + '=' * 80)
print(f'Pearson Correlation between LHS and Real Badcase Gene Vulnerability: {corr:.4f}')
print('=' * 80)

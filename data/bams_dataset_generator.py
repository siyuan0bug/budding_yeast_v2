#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAMS (Bifurcation-Aware Manifold Sampling) Dataset Generator
================================================================================
Canonical BAMS Generator (aligned with mutant_aware.py v5.2 biophysics)

Core Algorithmic Innovations:
  [Pillar 1] Kirchhoff Signed Flux-Balance Walk (基尔霍夫符号流守恒遍历):
             Zero hardcoded templates. Operates directly on the 54 signed directed
             edges of Chen (2004) ODE. Starts from an unbiased anchor gene
             g0 ~ Uniform(27 modules), traverses the graph, and assigns perturbation
             polarities according to Kirchhoff sign conservation:
               - Synthetic interactions / epistatic defects (LOF + LOF)
               - Genetic suppression / dosage compensation (LOF + downstream/parallel GOF)
               - Antagonistic checkpoint bypass (LOF + LOF on negative edges)

  [Pillar 2] Closed-Loop Numerical Bifurcation Bisection (闭环数值分岔夹逼):
             For candidate synthetic lethality + dosage suppression pairs, runs a
             fast 3-step ODE bisection on the parallel worker to pinpoint the
             critical dosage alpha* separating arrest (Pattern_C) from division (Pattern_A/B).
             Clusters samples tightly across the separatrix:
               alpha in {alpha* x 0.85, alpha* x 0.95, alpha* x 1.05, alpha* x 1.20}.
             Provides training trajectories right on the knife-edge of bifurcation.

  [Pillar 3] Coupled Biophysical Dynamic Range (双曲饱和动力学连续参数分布):
             100% faithful to mutant_aware.py Section 2.3.1 audited physical envelopes:
             GAL-OE promoter transcription, discrete multi-copy integers {2..30},
             tightened ts anchor envelopes, and destruction box <=1% buffers.

Zero-Leakage & Peer-Review Compliance:
  - Preserves stratified Level quotas (L1: 35%, L2: 30%, L3: 22%, L4: 13%).
  - L1 uses Latin-square round-robin grid rotation across all 27 genes x modes.
  - Zero hardcoding of Table S5 mutant strains or names.
  - Complete 141-D deduplication against all 126 real mutants.

Export Alignment:
  Matches exact .npz structure of mass_100000_dataset & mutant_aware_100000:
  - data           : (N, 39, 500) float32 array
  - params         : (N, 141) float64 array
  - param_names    : (141,) string array
  - mutant_names   : (N,) string array (126 real mutants first, then BAMS synthetic)
  - labels         : (N,) biological cell viability labels
  - pattern_labels : (N,) visual mass growth patterns
"""

import os
import sys
import math
import time
import signal
import warnings
import argparse
import random
import numpy as np
import concurrent.futures
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

warnings.filterwarnings("ignore")

# ==============================================================================
# Part 1: Direct Import of Master Biological Engine & Rules from mutant_aware.py
# (Single Source of Truth: eliminates duplication, ensures 100% biophysical consistency)
# ==============================================================================
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from data.mutant_aware import (
        # 39 ODE State Variables
        nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5, nSIC1P, nC2P, nC5P, 
        nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T, nCDC20, nCDH1T, nCDH1, nCDC14T, 
        nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, 
        nVi20, nlte1, nBUB2,
        # Parameters, Conditions & Keys
        get_default_params, PARAM_KEYS, initial_conds,
        # ODE Engine, GK ultra-sensitivity switch & Events
        GK, eqns, event1, event2, event3, event4,
        # Phenotype Analysis & Simulation Engine
        analyze_cell_viability_unified, analyze_mass_pattern, simulate_mutant,
        # Benchmark Dict & Deduplication
        REAL_MUTANTS, get_cached_real_params, is_duplicate_real_mutant,
        # Biological Modules & Regulatory Networks
        BINDING_GROUPS, GROUP_BY_NAME, GENE_NAMES,
        # Biophysical Continuous Sampler & Mutation Operators
        sample_scaling_factor, apply_gene_mutation
    )
except ImportError:
    from mutant_aware import (
        nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5, nSIC1P, nC2P, nC5P, 
        nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T, nCDC20, nCDH1T, nCDH1, nCDC14T, 
        nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, 
        nVi20, nlte1, nBUB2,
        get_default_params, PARAM_KEYS, initial_conds,
        GK, eqns, event1, event2, event3, event4,
        analyze_cell_viability_unified, analyze_mass_pattern, simulate_mutant,
        REAL_MUTANTS, get_cached_real_params, is_duplicate_real_mutant,
        BINDING_GROUPS, GROUP_BY_NAME, GENE_NAMES,
        sample_scaling_factor, apply_gene_mutation
    )

p_wt_default = get_default_params()

MODULE_EDGES = [
    ('CLN3','SBF_MBF','+'), ('BCK2','SBF_MBF','+'), ('CLN2','SBF_MBF','+'),
    ('CLB5','SBF_MBF','+'), ('CLB2','SBF_MBF','-'),
    ('SBF_MBF','CLN2','+'), ('SBF_MBF','CLB5','+'), ('SBF_MBF','CDC6','+'),
    ('CLB2','Mcm1','+'), ('Mcm1','CLB2','+'), ('Mcm1','SWI5','+'),
    ('Mcm1','CDC20','+'), ('Mcm1','PDS1','+'),
    ('CLB2','CDC20','+'), ('CDC20','CLB2','-'), ('CDC20','CLB5','-'),
    ('CDC20','PDS1','-'),
    ('CDH1','CLB2','-'), ('CDH1','PDS1','-'),
    ('CDC14','CDH1','+'), ('CDC14','SIC1','+'), ('CDC14','CDC6','+'),
    ('CLB2','CDH1','-'), ('CLB5','CDH1','-'), ('CLN2','CDH1','-'),
    ('CLN3','CDH1','-'),
    ('SIC1','CLB2','-'), ('SIC1','CLB5','-'),
    ('CDC6','CLB2','-'), ('CDC6','CLB5','-'),
    ('SWI5','SIC1','+'), ('SWI5','CDC6','+'),
    ('NET1','CDC14','-'), ('PPX','NET1','+'), ('CDC15','NET1','+'),
    ('TEM1','CDC15','+'), ('BUB2','TEM1','-'), ('MAD2','CDC20','-'),
    ('PDS1','ESP1','-'),
    ('CDC28','SIC1','+'), ('CDC28','CDC6','+'), ('CDC28','CDH1','+'),
    ('CDC28','SBF_MBF','+'), ('CDC28','ORI_BUD_SPN','+'),
    ('CLB2','ORI_BUD_SPN','+'), ('CLB5','ORI_BUD_SPN','+'),
    ('APC_Core','CLB2','-'), ('APC_Core','PDS1','-'),
    ('CDC20','APC_Core','+'), ('CDH1','APC_Core','+'),
    ('CLB2','SWI5','-'), ('CLB2','SIC1','-'), ('CLB5','SIC1','-'), ('CLN2','SIC1','-'),
]

ADJ = {n: set() for n in GENE_NAMES}
EDGES_OUT = {n: [] for n in GENE_NAMES}
EDGES_IN = {n: [] for n in GENE_NAMES}

for u, v, s in MODULE_EDGES:
    ADJ[u].add(v); ADJ[v].add(u)
    EDGES_OUT[u].append((v, s))
    EDGES_IN[v].append((u, s))

CONVERGENT_TARGETS = {}
for i, a in enumerate(GENE_NAMES):
    for b in GENE_NAMES[i+1:]:
        t_a = {v for v, _ in EDGES_OUT[a]}
        t_b = {v for v, _ in EDGES_OUT[b]}
        common = t_a & t_b
        if common:
            CONVERGENT_TARGETS[(a, b)] = sorted(common)

ANTAG_PAIRS = [(u, v) for u, v, s in MODULE_EDGES if s == '-']

LOF_MODES = {'KO', 'ts', 'DAmP', 'pLOF', 'db', 'cdc24Δ', 'cdc45Δ', 'spindle_defect'}
GOF_MODES = {'GAL-OE', 'MC', 'CA', 'APC-A', 'TAB6-1'}

def get_mode_polarity(mode):
    if mode in LOF_MODES: return -1
    if mode in GOF_MODES: return +1
    return 0

# ==============================================================================

def touch_set(g_name, m_chosen):
    ovr = {}
    p_wt = get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    apply_gene_mutation(g_name, m_chosen, 1.0, p_wt, {}, ovr)
    return set(ovr.keys())

def resolve_pair_conflicts(pairs):
    pairs = list(pairs)
    rounds = 20
    while len(pairs) > 1 and rounds > 0:
        rounds -= 1
        touched = {}
        conflict = None
        for i, (g, m) in enumerate(pairs):
            ts = touch_set(g, m)
            for pk in ts:
                if pk in touched:
                    conflict = (i, touched[pk])
                    break
                touched[pk] = i
            if conflict is not None:
                break
        if conflict is None:
            break
        i, j = conflict
        gj = pairs[j][0]
        modes_j = list(GROUP_BY_NAME[gj]['modes'].keys())
        pairs[j] = (gj, str(np.random.choice(modes_j)))
    return pairs

# ==============================================================================
# Part 5: Pillar 2 - Closed-Loop Numerical Bifurcation Bisection
# ==============================================================================

def fast_eval_division_status(ovr, tf=180):
    """
    Lightweight ODE execution (~12ms) to check whether cell divides or arrests.
    Returns: True if cell divides (Pattern_A/B), False if arrested (Pattern_C).
    """
    _, yu, _, pat = simulate_mutant("bisect_probe", ovr, tf=tf, ns=150)
    if yu is None or pat == "Pattern_C: Exponential-Arrest (==0)":
        return False
    return True

def bisect_bifurcation_dosage(base_ovr, target_gene, mode, alpha_min, alpha_max, n_steps=3):
    """
    3-step log-scale numerical bisection on ODE solver.
    Locates the knife-edge threshold alpha* where the network transitions
    between arrest and division.
    """
    p_wt = get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

    ovr_lo = dict(base_ovr)
    apply_gene_mutation(target_gene, mode, alpha_min, p_wt, {}, ovr_lo)
    div_lo = fast_eval_division_status(ovr_lo)

    ovr_hi = dict(base_ovr)
    apply_gene_mutation(target_gene, mode, alpha_max, p_wt, {}, ovr_hi)
    div_hi = fast_eval_division_status(ovr_hi)

    if div_lo == div_hi:
        return None

    lo, hi = alpha_min, alpha_max
    for _ in range(n_steps):
        mid = np.sqrt(lo * hi)
        ovr_mid = dict(base_ovr)
        apply_gene_mutation(target_gene, mode, mid, p_wt, {}, ovr_mid)
        div_mid = fast_eval_division_status(ovr_mid)
        if div_mid == div_lo:
            lo = mid
        else:
            hi = mid

    return np.sqrt(lo * hi)

# ==============================================================================
# Part 6: Pillar 1 - Kirchhoff Signed Flux-Balance Walk Sampler
# ==============================================================================

_L1_GRID = [(g, m) for g in GENE_NAMES for m in GROUP_BY_NAME[g]['modes']]
np.random.RandomState(20260831).shuffle(_L1_GRID)
_L1_IDX = {'i': 0}

def get_next_L1_cell():
    cell = _L1_GRID[_L1_IDX['i'] % len(_L1_GRID)]
    _L1_IDX['i'] += 1
    return cell

def sample_bams_skeleton_and_modes(level):
    """
    Samples multi-gene skeleton and coordinated modes based on signed topology.
    Level:
      1: L1 grid rotation (1 gene)
      2: Direct topological edge coupling (2 genes)
      3: Flux-balance motif walk (3~5 genes)
      4: Cross-pathway extended walk (4~6 genes)
    """
    if level == 1:
        g, m = get_next_L1_cell()
        return [(g, m)], 'L1_Grid', None

    g0 = str(np.random.choice(GENE_NAMES))
    modes0 = list(GROUP_BY_NAME[g0]['modes'].keys())
    m0 = str(np.random.choice(modes0))
    s0 = get_mode_polarity(m0)
    if s0 == 0: s0 = -1

    pairs = [(g0, m0)]
    bisection_spec = None
    strat_desc = ""

    if level == 2:
        sub_type = np.random.choice(['antag', 'cascade', 'convergent', 'random'], p=[0.35, 0.35, 0.20, 0.10])
        strat_desc = f"L2_{sub_type}"

        if sub_type == 'antag' and ANTAG_PAIRS:
            cand_antag = [v for u, v, s in MODULE_EDGES if u == g0 and s == '-'] + \
                         [u for u, v, s in MODULE_EDGES if v == g0 and s == '-']
            if cand_antag:
                g1 = str(np.random.choice(cand_antag))
            else:
                u, v = ANTAG_PAIRS[np.random.choice(len(ANTAG_PAIRS))]
                g0, g1 = u, v
                m0 = 'KO' if 'KO' in GROUP_BY_NAME[g0]['modes'] else list(GROUP_BY_NAME[g0]['modes'].keys())[0]
                pairs[0] = (g0, m0)
            
            m1_pool = [m for m in ['KO', 'ts', 'db', 'GAL-OE', 'MC'] if m in GROUP_BY_NAME[g1]['modes']]
            m1 = str(np.random.choice(m1_pool)) if m1_pool else list(GROUP_BY_NAME[g1]['modes'].keys())[0]
            pairs.append((g1, m1))

        elif sub_type == 'cascade':
            neighbors = [(v, s) for v, s in EDGES_OUT[g0]] + [(u, s) for u, s in EDGES_IN[g0]]
            if neighbors:
                g1, sign = neighbors[np.random.choice(len(neighbors))]
                if s0 == -1 and sign == '+':
                    m1_pool = [m for m in ['GAL-OE', 'MC'] if m in GROUP_BY_NAME[g1]['modes']]
                    m1 = str(np.random.choice(m1_pool)) if m1_pool else 'KO'
                    if m1 in ['MC', 'GAL-OE']:
                        bisection_spec = (g1, m1, 1.0 if m1=='MC' else 0.01, 30.0 if m1=='MC' else 12.0)
                else:
                    m1_pool = list(GROUP_BY_NAME[g1]['modes'].keys())
                    m1 = str(np.random.choice(m1_pool))
                pairs.append((g1, m1))
            else:
                g1 = str(np.random.choice([g for g in GENE_NAMES if g != g0]))
                pairs.append((g1, str(np.random.choice(list(GROUP_BY_NAME[g1]['modes'].keys())))))

        elif sub_type == 'convergent' and CONVERGENT_TARGETS:
            matching_pairs = [(a, b) for (a, b) in CONVERGENT_TARGETS.keys() if a == g0 or b == g0]
            if matching_pairs:
                a, b = matching_pairs[np.random.choice(len(matching_pairs))]
                g1 = b if a == g0 else a
            else:
                (a, b) = list(CONVERGENT_TARGETS.keys())[np.random.choice(len(CONVERGENT_TARGETS))]
                g0, g1 = a, b
                pairs[0] = (g0, 'KO' if 'KO' in GROUP_BY_NAME[g0]['modes'] else list(GROUP_BY_NAME[g0]['modes'].keys())[0])
            
            m1_lofs = [m for m in ['KO', 'ts', 'db'] if m in GROUP_BY_NAME[g1]['modes']]
            m1 = str(np.random.choice(m1_lofs)) if m1_lofs else list(GROUP_BY_NAME[g1]['modes'].keys())[0]
            pairs.append((g1, m1))

        else:
            g1 = str(np.random.choice([g for g in GENE_NAMES if g != g0]))
            pairs.append((g1, str(np.random.choice(list(GROUP_BY_NAME[g1]['modes'].keys())))))

    else:
        ki = int(np.random.randint(3, 6)) if level == 3 else int(np.random.randint(4, 7))
        strat_desc = f"L{level}_FluxWalk"

        visited = {g0}
        walk_history = [(g0, s0)]

        if s0 == 1 and np.random.rand() < 0.70:
            lofs = [m for m in ['KO', 'ts', 'db'] if m in GROUP_BY_NAME[g0]['modes']]
            if lofs:
                m0 = 'KO' if 'KO' in lofs else lofs[0]
                pairs[0] = (g0, m0)
                s0 = -1
                walk_history[0] = (g0, s0)

        while len(pairs) < ki:
            frontier = []
            for u in visited:
                for v, s in EDGES_OUT[u]:
                    if v not in visited: frontier.append((u, v, s, 'out'))
                for v, s in EDGES_IN[u]:
                    if v not in visited: frontier.append((u, v, s, 'in'))

            if frontier:
                u, v, sign, direction = frontier[np.random.choice(len(frontier))]
                visited.add(v)
                u_polar = next((pol for g_u, pol in walk_history if g_u == u), -1)
                
                is_compensation = (np.random.rand() < 0.60)
                if is_compensation and u_polar == -1 and sign == '+':
                    oes = [m for m in ['MC', 'GAL-OE'] if m in GROUP_BY_NAME[v]['modes']]
                    if oes:
                        mv = str(np.random.choice(oes))
                        if bisection_spec is None:
                            bisection_spec = (v, mv, 1.0 if mv=='MC' else 0.01, 30.0 if mv=='MC' else 12.0)
                    else:
                        mv = str(np.random.choice(list(GROUP_BY_NAME[v]['modes'].keys())))
                elif is_compensation and u_polar == -1 and sign == '-':
                    lofs = [m for m in ['KO', 'ts', 'db'] if m in GROUP_BY_NAME[v]['modes']]
                    mv = str(np.random.choice(lofs)) if lofs else list(GROUP_BY_NAME[v]['modes'].keys())[0]
                else:
                    mv = str(np.random.choice(list(GROUP_BY_NAME[v]['modes'].keys())))
                
                v_polar = get_mode_polarity(mv)
                walk_history.append((v, v_polar))
                pairs.append((v, mv))
            else:
                unvisited = [g for g in GENE_NAMES if g not in visited]
                if not unvisited: break
                v = str(np.random.choice(unvisited))
                visited.add(v)
                mv = str(np.random.choice(list(GROUP_BY_NAME[v]['modes'].keys())))
                pairs.append((v, mv))

    # Same-gene composite: expression x structure dual mode, 25% (matching mutant_aware.py)
    COMPOSITE_ELIGIBLE = [n for n in GENE_NAMES
                          if any(m in GROUP_BY_NAME[n]['modes'] for m in ('GAL-OE', 'MC'))
                          and any(m in GROUP_BY_NAME[n]['modes'] for m in ('db', 'PM', 'CA'))]
    expanded = []
    for g, m in pairs:
        expanded.append((g, m))
        gmode = GROUP_BY_NAME.get(g, {}).get('modes', {})
        if g in COMPOSITE_ELIGIBLE and np.random.rand() < 0.25:
            synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
            struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
            if m in synth and struc:
                expanded.append((g, str(np.random.choice(struc))))
            elif m in struc and synth:
                expanded.append((g, str(np.random.choice(synth))))
    pairs = resolve_pair_conflicts(expanded)
    return pairs, strat_desc, bisection_spec

def rescue_collision_candidate(ovr, desc_str, p_wt):
    """
    Rescues a candidate sample that collided with a real mutant:
    70%: Continuous mode shift (DAmP / ts / pLOF) on Glucose.
    30%: Carbon shift to Galactose (150) or Raffinose (160).
    """
    tokens = desc_str.split('+')
    parsed = []
    for tok in tokens:
        if '[' in tok and tok.endswith(']'):
            parts = tok.split('[')
            parsed.append((parts[0], parts[1][:-1]))
        else:
            parsed.append((tok, 'KO'))

    if not parsed:
        return None, desc_str

    p_mut = p_wt.copy()
    new_ovr = {}

    if np.random.rand() < 0.70:
        ko_genes = [g for g, m in parsed if m == 'KO' and g in GROUP_BY_NAME]
        if ko_genes:
            target_g = str(np.random.choice(ko_genes))
            avail = [m for m in ['DAmP', 'ts', 'pLOF'] if m in GROUP_BY_NAME[target_g]['modes']]
            if avail:
                target_m = str(np.random.choice(avail))
                new_pairs = [(g, target_m if g == target_g else m) for g, m in parsed]
                new_desc = []
                for g_name, m_chosen in new_pairs:
                    new_desc.append(f"{g_name}[{m_chosen}]")
                    alpha = sample_scaling_factor(m_chosen, g_name) \
                        if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
                    apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, new_ovr)
                new_ovr['mdt'] = ovr.get('mdt', 90.0)
                return new_ovr, "+".join(new_desc)

    shift_mdt = float(np.random.choice([150.0, 160.0]))
    shifted_ovr = dict(ovr)
    shifted_ovr['mdt'] = shift_mdt
    return shifted_ovr, desc_str + f"_MDT[{int(shift_mdt)}]"

def sample_bams_mutant_task(level=2, defaults=None):
    """
    Fast task constructor (Phase 2). Returns initial overrides, descriptor, and optional bisection spec.
    """
    if defaults is None:
        defaults = get_default_params()
        defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

    p_wt = defaults
    ovr = {}
    pairs, strat_desc, bisection_spec = sample_bams_skeleton_and_modes(level)

    desc_parts = []
    has_gal = False

    for g, m in pairs:
        if m == 'GAL-OE': has_gal = True
        alpha = sample_scaling_factor(m, g)
        apply_gene_mutation(g, m, alpha, p_wt, {}, ovr)
        desc_parts.append(f"{g}[{m}]")

    if has_gal:
        ovr['mdt'] = 150.0
    else:
        ovr['mdt'] = float(np.random.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))

    desc_str = "+".join(desc_parts) + f"_{strat_desc}"
    return ovr, desc_str, bisection_spec

# ==============================================================================
# Worker Task Simulation with Parallel Bifurcation Bisection
# ==============================================================================

def execute_simulation_task(task_spec):
    """
    Executed inside each worker process:
    1. If bisection_spec is attached, runs 3-step ODE bisection to cluster dosage around alpha*.
    2. Simulates full 210 min / 500 steps trajectory.
    """
    name, ovr, bisection_spec, tf, ns = task_spec
    defaults = get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

    if bisection_spec is not None:
        target_g, target_m, a_min, a_max = bisection_spec
        base_ovr = dict(ovr)
        try:
            alpha_star = bisect_bifurcation_dosage(base_ovr, target_g, target_m, a_min, a_max, n_steps=3)
            if alpha_star is not None:
                jitter = float(np.random.choice([0.85, 0.95, 1.05, 1.20]))
                refined_alpha = float(np.clip(alpha_star * jitter, a_min, a_max))
                apply_gene_mutation(target_g, target_m, refined_alpha, defaults, {}, ovr)
        except Exception:
            pass

    return simulate_mutant(name, ovr, tf=tf, ns=ns), ovr

# ==============================================================================
# Part 7: Complete Dataset Generation & .npz Export Engine
# ==============================================================================

def generate_bams_dataset(n_synthetic=100000, level_ratios=[0.35, 0.30, 0.22, 0.13],
                          max_attempts=None, num_workers=None,
                          output_name=None, output_dir=None, enable_bisection=True,
                          seed=42):
    """
    Generates complete BAMS dataset (126 Real Mutants + n_synthetic Synthetic Mutants)
    with exact schema matching mass_100000_dataset & mutant_aware_100000.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    t_init = time.time()
    if output_name is None:
        output_name = f"bams_{n_synthetic}_210m_500s.npz"

    if output_dir is None:
        if os.path.exists("/home/users/zsy/mydata/budding_yeast"):
            output_dir = "/home/users/zsy/mydata/budding_yeast"
        else:
            output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, output_name)

    print("=" * 85, flush=True)
    print(" BAMS (Bifurcation-Aware Manifold Sampling) DATASET GENERATOR", flush=True)
    print(f" Target Output Path : {dataset_path}", flush=True)
    print(f" Synthetic Samples  : {n_synthetic}", flush=True)
    print(f" Level Ratios       : L1={level_ratios[0]} | L2={level_ratios[1]} | L3={level_ratios[2]} | L4={level_ratios[3]}", flush=True)
    print(f" Bifurcation Bisect : {enable_bisection}", flush=True)
    print("=" * 85, flush=True)

    if max_attempts is None:
        max_attempts = max(50000, n_synthetic * 10)

    defaults = get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = PARAM_KEYS

    simulation_tasks = []
    ordered_names = []

    # Phase 1: Load 126 Real Mutants (Appendix B Table S5)
    print(f"\n[Phase 1/4] Preparing D_real (126 Real Mutants from Appendix B Table S5)...", flush=True)
    for mid, (mname, movr) in REAL_MUTANTS.items():
        sample_name = f"{mid}_{mname}".replace(" ", "_")
        ordered_names.append(sample_name)
        simulation_tasks.append((sample_name, movr, None, 210, 500))
    print(f" -> Loaded {len(REAL_MUTANTS)} Real Mutants.", flush=True)

    # Phase 2: Sample Synthetic Tasks (BAMS Protocol)
    print(f"\n[Phase 2/4] Sampling {n_synthetic} Synthetic Tasks (BAMS Protocol)...", flush=True)
    r1, r2, r3, r4 = level_ratios
    tot = r1 + r2 + r3 + r4
    r1, r2, r3, r4 = r1/tot, r2/tot, r3/tot, r4/tot

    n_l1 = int(math.floor(r1 * n_synthetic))
    n_l2 = int(math.floor(r2 * n_synthetic))
    n_l3 = int(math.floor(r3 * n_synthetic))
    n_l4 = n_synthetic - n_l1 - n_l2 - n_l3

    print(f" -> Quota: L1={n_l1} ({r1*100:.1f}%) | L2={n_l2} ({r2*100:.1f}%) | L3={n_l3} ({r3*100:.1f}%) | L4={n_l4} ({r4*100:.1f}%)", flush=True)

    level_assignments = [1]*n_l1 + [2]*n_l2 + [3]*n_l3 + [4]*n_l4
    np.random.shuffle(level_assignments)

    attempts = 0
    synth_success = 0
    blocked_dedup = 0
    rescued_dedup = 0

    while synth_success < n_synthetic and attempts < max_attempts:
        attempts += 1
        lvl = level_assignments[synth_success]
        ovr, desc_str, bisect_spec = sample_bams_mutant_task(level=lvl, defaults=defaults)

        if is_duplicate_real_mutant(ovr, defaults):
            rescued_ovr, rescued_desc = rescue_collision_candidate(ovr, desc_str, defaults)
            if rescued_ovr is not None and not is_duplicate_real_mutant(rescued_ovr, defaults):
                ovr = rescued_ovr
                desc_str = rescued_desc
                bisect_spec = None
                rescued_dedup += 1
            else:
                blocked_dedup += 1
                continue

        synth_success += 1
        sample_name = f"BAMS_{desc_str}_L{lvl}_{synth_success}".replace(" ", "_")
        ordered_names.append(sample_name)
        active_bisect = bisect_spec if enable_bisection else None
        simulation_tasks.append((sample_name, ovr, active_bisect, 210, 500))

    print(f" -> Sampling Completed: {synth_success} synthetic tasks generated in {attempts} attempts ({time.time()-t_init:.2f}s).", flush=True)
    print(f"    - Deduplication: {blocked_dedup} candidates blocked, {rescued_dedup} candidates rescued into continuous margin.", flush=True)
    print(f" -> Total Simulation Queue: {len(simulation_tasks)} tasks ({len(REAL_MUTANTS)} Real + {synth_success} Synthetic).", flush=True)

    # Phase 3: Parallel ODE Simulation & Bisection
    if num_workers is None:
        total_cores = os.cpu_count() or 4
        num_workers = min(60, total_cores)

    print(f"\n[Phase 3/4] Simulating {len(simulation_tasks)} Mutant Trajectories on {num_workers} CPU Workers...", flush=True)
    sim_counts = {'success': 0, 'discarded': 0, 'fail': 0}
    results = {}
    bio_labels = {}
    pattern_labels = {}
    final_task_ovrs = {}

    log_step = max(50, len(simulation_tasks) // 40)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(execute_simulation_task, task): task[0]
            for task in simulation_tasks
        }
        for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):
            name = future_to_task[future]
            try:
                (m_name, tensor, bio_l, pat_l), refined_ovr = future.result()
                if tensor is not None:
                    sim_counts['success'] += 1
                    results[m_name] = tensor
                    bio_labels[m_name] = bio_l
                    pattern_labels[m_name] = pat_l
                    final_task_ovrs[m_name] = refined_ovr
                else:
                    if bio_l == "Discarded_Mass":
                        sim_counts['discarded'] += 1
                    else:
                        sim_counts['fail'] += 1
            except Exception:
                sim_counts['fail'] += 1

            if (i + 1) % log_step == 0 or (i + 1) == len(simulation_tasks):
                pct = (i + 1) / len(simulation_tasks) * 100
                elapsed = time.time() - t_init
                print(f"  [{pct:5.1f}%] Tasks: {i+1}/{len(simulation_tasks)} | Success: {sim_counts['success']} | Discarded: {sim_counts['discarded']} | Fail: {sim_counts['fail']} | Elapsed: {elapsed:.1f}s", flush=True)

    # Phase 4: Format Alignment & Export .npz
    print(f"\n[Phase 4/4] Aligning Parameter Matrices & Exporting to {dataset_path}...", flush=True)
    ordered_tensors = []
    final_names = []
    final_params = []
    final_bio_labels = []
    final_pattern_labels = []

    for name in ordered_names:
        if name in results:
            ordered_tensors.append(results[name])
            final_names.append(name)
            final_bio_labels.append(bio_labels[name])
            final_pattern_labels.append(pattern_labels[name])
            ovr = final_task_ovrs[name]
            p_vec = [ovr.get(k, defaults[k]) for k in param_keys]
            final_params.append(p_vec)

    all_tensors = np.array(ordered_tensors, dtype=np.float32)
    all_params = np.array(final_params, dtype=np.float64)
    all_param_names = np.array(param_keys, dtype=str)
    all_mutant_names = np.array(final_names, dtype=str)
    all_bio_labels = np.array(final_bio_labels, dtype=str)
    all_pattern_labels = np.array(final_pattern_labels, dtype=str)

    np.savez_compressed(
        dataset_path,
        data=all_tensors,
        params=all_params,
        param_names=all_param_names,
        mutant_names=all_mutant_names,
        labels=all_bio_labels,
        pattern_labels=all_pattern_labels
    )

    t_total = time.time() - t_init
    print("=" * 85, flush=True)
    print(f" BAMS DATASET GENERATION COMPLETE IN {t_total:.1f} SECONDS", flush=True)
    print("=" * 85, flush=True)
    print(f"Dataset Export Path               : {dataset_path}", flush=True)
    print(f"Data Tensor Shape (N, Vars, Steps): {all_tensors.shape}", flush=True)
    print(f"Params Matrix Shape (N, P_dim)    : {all_params.shape}", flush=True)
    print(f"Total Validated Samples Saved     : {len(final_names)} / {len(ordered_names)} ({len(final_names)/len(ordered_names)*100:.2f}%)", flush=True)
    print("=" * 85, flush=True)
    return dataset_path

def main():
    parser = argparse.ArgumentParser(description="BAMS (Bifurcation-Aware Manifold Sampling) Generator")
    parser.add_argument("--n_synthetic", type=int, default=100000, help="Number of synthetic samples to generate")
    parser.add_argument("--output_name", type=str, default=None, help="Name of export .npz file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save export .npz")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU workers for parallel simulation")
    parser.add_argument("--level_ratios", nargs=4, type=float, default=[0.35, 0.30, 0.22, 0.13],
                        help="Quotas for Level 1, 2, 3, 4 (default: 0.35 0.30 0.22 0.13)")
    parser.add_argument("--no_bisection", action="store_true", help="Disable closed-loop bifurcation bisection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_bams_dataset(
        n_synthetic=args.n_synthetic,
        level_ratios=args.level_ratios,
        num_workers=args.workers,
        output_name=args.output_name,
        output_dir=args.output_dir,
        enable_bisection=(not args.no_bisection),
        seed=args.seed
    )

if __name__ == "__main__":
    main()

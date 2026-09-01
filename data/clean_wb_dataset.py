#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WT-Blind 数据集清洗脚本
=======================
问题根因: mass_wt_blind_sampler.py 的 GAL-OE 自洽反演 (_gal_theta_and_invert)
在多重突变叠加时数值不稳定, 产生的过表达倍数 alpha 无上限约束,
导致 ~1.7% 样本的 GAL-OE 类参数 (ksb2p/ksn2p/C0/ksf6p/...) 达到 1e8~1e28。

后果: 数据集级 p_std 被拉到 ~1e23, 99.8% 正常样本的标准化参数被压缩到 ~0,
模型完全无法区分不同参数 —— 这是 wb 数据集上所有模型 relL2 都很差
(unet1d 0.49 vs random 数据集 0.19) 的元凶。

清洗规则 (保守, 不覆盖原文件):
  1. 剔除轨迹含 NaN/Inf 的样本
  2. 剔除任意 |参数| > P_THRESH (默认 1e3) 的样本
     (GAL-OE 物理上不超过默认值 1000 倍; 反演产生的 1e8~1e28 值必须整样本剔除,
      其轨迹也是非物理参数仿真的产物)
  3. 剔除轨迹绝对值 > Y_THRESH (默认 1e4) 的样本
     (组合突变的 ODE 动力学可自行爆炸: 实测 wb 轨迹最大 8.3e268 vs random 8.9e3;
      这些 outlier 会污染 mean_y/std_y 归一化统计量并主导 MSE 梯度)

用法:
  python clean_wb_dataset.py [--input <npz>] [--output <npz>] [--p_thresh 1e3] [--y_thresh 1e4]
"""
import argparse
import numpy as np

DEFAULT_IN = '/home/users/zsy/mydata/budding_yeast/wb_mass_100000_dataset_210min_500steps_dual_labels.npz'
DEFAULT_OUT = '/home/users/zsy/mydata/budding_yeast/wb_mass_100000_clean_dataset_210min_500steps_dual_labels.npz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default=DEFAULT_IN)
    ap.add_argument('--output', type=str, default=DEFAULT_OUT)
    ap.add_argument('--p_thresh', type=float, default=1e3)
    ap.add_argument('--y_thresh', type=float, default=1e4)
    args = ap.parse_args()

    d = np.load(args.input, allow_pickle=True)
    y, prm = d['data'], d['params']
    names = d['mutant_names']
    N = len(y)
    print(f'原始样本数: {N}')

    # 规则 1: 轨迹非有限
    bad_y = ~np.isfinite(y).all(axis=(1, 2))
    # 规则 2: 参数异常
    bad_p = (np.abs(prm) > args.p_thresh).any(axis=1)
    # 规则 3: 轨迹爆炸
    bad_traj = np.abs(y).max(axis=(1, 2)) > args.y_thresh
    keep = ~(bad_y | bad_p | bad_traj)
    print(f'剔除: 轨迹非有限 {bad_y.sum()} + 参数异常(|p|>{args.p_thresh:g}) {bad_p.sum()} '
          f'+ 轨迹爆炸(|y|>{args.y_thresh:g}) {bad_traj.sum()}')
    print(f'保留: {keep.sum()} 个 ({keep.mean():.2%})')

    # 清洗前后参数统计对比 (p_std 是模型标准化的关键)
    pstd_before = prm.std(axis=0)
    pstd_after = prm[keep].std(axis=0)
    print(f'\n参数 std 均值: 清洗前 {pstd_before.mean():.3e} -> 清洗后 {pstd_after.mean():.3e}')
    print(f'参数 |max|:    清洗前 {np.abs(prm).max():.3e} -> 清洗后 {np.abs(prm[keep]).max():.3e}')
    # 轨迹统计 (mean_y/std_y 同样被 outlier 污染)
    ymax_before = np.abs(y).max()
    ymax_after = np.abs(y[keep]).max()
    print(f'轨迹 |max|:   清洗前 {ymax_before:.3e} -> 清洗后 {ymax_after:.3e}')

    # 异常样本的突变构成 top (信息记录)
    removed = names[~keep]
    from collections import Counter
    genes = Counter()
    for nm in removed:
        for tok in str(nm).split('+'):
            genes[tok.split('[')[0].replace('WB-MASS_', '')] += 1
    print('\n剔除样本的基因构成 top10:', dict(genes.most_common(10)))

    # 保存 (仅对第一维为样本数的数组做子集)
    out = {}
    for k, v in d.items():
        arr = np.asarray(v)
        if arr.ndim > 0 and arr.shape and arr.shape[0] == N:
            out[k] = arr[keep]
        else:
            out[k] = arr
    np.savez_compressed(args.output, **out)
    print(f'\n已保存清洗后数据集: {args.output}')

    # 验证
    d2 = np.load(args.output, allow_pickle=True)
    assert np.isfinite(d2['data']).all(), '清洗后仍有非有限轨迹!'
    assert (np.abs(d2['params']) <= args.p_thresh).all(), '清洗后仍有异常参数!'
    assert np.abs(d2['data']).max() <= args.y_thresh, '清洗后仍有爆炸轨迹!'
    print(f'验证通过: {d2["data"].shape[0]} 个样本全部干净')


if __name__ == '__main__':
    main()

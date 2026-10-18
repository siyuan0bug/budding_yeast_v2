# Pure-random real-mutant bad-case diagnosis

Reference run: `/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/triple_dynamic_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_mutant_metrics.csv`  
Models included: 47; real mutants: 126.

Primary diagnostic rule: MAE > 0.15, MSE > 1.0, Relative L2 > 0.25, or correlation < 0.75.
These are descriptive thresholds, not training or model-selection criteria. Identical result files are counted once.
 It flags **22** mutants; **51** are bad in at least half of the evaluated architectures.

## Persistent joint bad cases

| Mutant | Pattern | MAE | MSE | relL2 | corr | bad models | exact coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 22_cln1Δ_cln2Δ_cln3Δ_multi-copy_CLB5 | Pattern_B: Single-Division (==1) | 0.5708 | 32.0829 | 0.5078 | 0.4658 | 100% | unknown |
| 20_cln1Δ_cln2Δ_cln3Δ_sic1Δ | Pattern_B: Single-Division (==1) | 0.2586 | 3.5483 | 0.4648 | 0.5207 | 100% | unknown |
| 76_cdc20Δ_pds1Δ_clb5Δ | Pattern_B: Single-Division (==1) | 0.1683 | 1.0101 | 0.4053 | 0.6668 | 100% | unknown |
| 23_cln1Δ_cln2Δ_cln3Δ_GAL-CLB5 | Pattern_C: Exponential-Arrest (==0) | 0.2115 | 4.1313 | 0.2707 | 0.6271 | 94% | unknown |
| 33_GAL-SIC1_GAL-CLN2_cln1Δ_cln2Δ_cdh1Δ | Pattern_B: Single-Division (==1) | 0.1824 | 1.8956 | 0.4238 | 0.5710 | 91% | unknown |
| 42_sic1Δ_cdc6Δ_2-49_cdh1Δ_GAL-CDC20 | Pattern_B: Single-Division (==1) | 0.2334 | 0.9476 | 0.3955 | 0.3515 | 100% | unknown |
| 37_sic1Δ_cdh1Δ_GAL-CDC20 | Pattern_C: Exponential-Arrest (==0) | 0.2080 | 1.4626 | 0.0698 | 0.7168 | 100% | unknown |
| 53_GAL-CLB2_sic1Δ | Pattern_B: Single-Division (==1) | 0.1450 | 1.1162 | 0.2939 | 0.7014 | 100% | unknown |
| 124_APC-A_GAL-CLB2 | Pattern_C: Exponential-Arrest (==0) | 0.9990 | 57.1893 | 0.1127 | 0.8545 | 100% | unknown |
| 62_GAL-CLB2-dbΔ | Pattern_C: Exponential-Arrest (==0) | 0.2690 | 3.7390 | 0.0566 | 0.8872 | 100% | unknown |
| 15_cln3Δ_bck2Δ_multi-copy_CLN2 | Pattern_C: Exponential-Arrest (==0) | 0.0715 | 0.1068 | 0.7041 | 0.5419 | 100% | unknown |
| 50_multi-copy_GAL-CLB2 | Pattern_C: Exponential-Arrest (==0) | 0.2383 | 2.9996 | 0.0156 | 0.8854 | 98% | unknown |
| 24_cln1Δ_cln2Δ_cln3Δ_multi-copy_BCK2 | Pattern_B: Single-Division (==1) | 0.0979 | 1.2993 | 0.2724 | 0.7715 | 91% | unknown |
| 71_CLB5-dbΔ_pds1Δ_cdc20Δ | Pattern_C: Exponential-Arrest (==0) | 0.1570 | 2.2777 | 0.0277 | 0.9170 | 89% | unknown |
| 59_CLB2-dbΔ_multi-copy_CDC6 | Pattern_B: Single-Division (==1) | 0.1457 | 1.7058 | 0.2521 | 0.8436 | 85% | unknown |
| 13_cln3Δ_bck2Δ | Pattern_C: Exponential-Arrest (==0) | 0.0070 | 0.0007 | 0.0302 | 0.7410 | 98% | unknown |
| 75_cdc20Δ_pds1Δ | Pattern_C: Exponential-Arrest (==0) | 0.1360 | 1.3413 | 0.0323 | 0.9158 | 96% | unknown |
| 44_swi5Δ_GAL-CLB2 | Pattern_C: Exponential-Arrest (==0) | 0.1627 | 0.5989 | 0.1268 | 0.8765 | 94% | unknown |
| 32_GAL-SIC1_cln1Δ_cln2Δ_cdh1Δ | Pattern_C: Exponential-Arrest (==0) | 0.0172 | 0.0065 | 0.0328 | 0.7498 | 94% | unknown |
| 58_CLB2-dbΔ_GAL-SIC1 | Pattern_B: Single-Division (==1) | 0.1269 | 1.3335 | 0.2408 | 0.8269 | 87% | unknown |
| 64_cln1Δ_cln2Δ_clb5Δ_clb6Δ | Pattern_C: Exponential-Arrest (==0) | 0.0104 | 0.0009 | 0.0560 | 0.7463 | 87% | unknown |
| 105_cdc14-ts_sic1Δ | Pattern_C: Exponential-Arrest (==0) | 0.1843 | 0.7578 | 0.2429 | 0.8115 | 60% | unknown |
| 46_swi5Δ_cdh1Δ_GAL-SIC1 | Pattern_B: Single-Division (==1) | 0.0769 | 0.1266 | 0.1532 | 0.7990 | 98% | unknown |
| 54_GAL-CLB2_cdh1Δ | Pattern_C: Exponential-Arrest (==0) | 0.1086 | 0.4106 | 0.0949 | 0.8248 | 96% | unknown |
| 41_sic1Δ_cdc6Δ_2-49_cdh1Δ | Pattern_C: Exponential-Arrest (==0) | 0.0629 | 0.0970 | 0.0681 | 0.7964 | 96% | unknown |
| 74_cdc20Δ_clb5Δ | Pattern_C: Exponential-Arrest (==0) | 0.0582 | 0.3070 | 0.0239 | 0.9109 | 94% | unknown |
| 122_APC-A_cdh1Δ_multi-copy_CDC20 | Pattern_A: Multi-Division (>=2) | 0.0759 | 0.3244 | 0.0960 | 0.8798 | 89% | unknown |
| 118_APC-A_cdh1Δ_multi-copy_SIC1 | Pattern_C: Exponential-Arrest (==0) | 0.0150 | 0.0061 | 0.0170 | 0.7529 | 89% | unknown |
| 66_GAL-CLB5_sic1Δ | Pattern_A: Multi-Division (>=2) | 0.1324 | 0.6209 | 0.1797 | 0.8471 | 87% | unknown |
| 30_GAL-SIC1_cln1Δ_cln2Δ | Pattern_C: Exponential-Arrest (==0) | 0.0082 | 0.0018 | 0.0100 | 0.7957 | 85% | unknown |

Coverage is not imported from archived MASS reports. If `--coverage` is supplied, the caller must verify that it was generated from this exact NPZ. Use `audit_pure_random_evidence.py` for direct train-split geometry evidence without a genotype-name parser.

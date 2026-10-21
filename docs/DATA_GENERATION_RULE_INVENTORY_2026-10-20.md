# 新旧文件与函数规则清单

> **当前路径（2026-10-21收口后）**：公共框架已迁入 `data/systems/{contracts,generation,dataset_io,registry}.py`；
> 配置在各系统recipes.py，采样实现在各系统samplers/。以下逐函数表保留历史路径/行号，
> 最新映射可运行已迁移的audit_rule_inventory.py；当前职责见 [整理说明](DATA_GENERATION_REFACTOR_2026-10-20.md)。


这是逐文件出处清单，不是“全部规则等价”的证明。默认值、全局绑定、生成主循环和导出规则另见 [逐文件结论](DATA_GENERATION_FILE_RULES_2026-10-20.md)。

> **路径说明（2026-10-21）**：下表 `data/sampling/...` 记录的是目录调整前的路径。现行实现位于
> `data/systems/budding/samplers/` 与 `data/systems/fission/samplers/`；文件内容与规则未变。

原文件 44 个，新 Python 文件 55 个。

## 原文件覆盖

| 原文件 | 范围 | 源码匹配定义数 | 需文字解释定义数 | 原文件 SHA256 |
| --- | --- | ---: | ---: | --- |
| `data/LHS/__init__.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v0.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1_origin.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1_origin_guaranteed.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1_origin_no_filter.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1_origin_v2.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1_origin_v3.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1_origin_v4.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1_origin_v5.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1_origin_v5_boundary.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/lhs_v1_origin_v6.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/LHS/test_lhs_filter_analysis.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/__init__.py` | consumer_migrated | 0 | 0 | 已变 |
| `data/al_pool_manager.py` | consumer_migrated | 0 | 0 | 已变 |
| `data/bams_dataset_generator.py` | migrated | 9 | 3 | 未变 |
| `data/build_bio_multirelational_graphs.py` | original_module_unchanged | 0 | 0 | 未变 |
| `data/clean_wb_dataset.py` | original_module_unchanged | 0 | 0 | 未变 |
| `data/dataset_utils.py` | consumer_migrated | 0 | 0 | 已变 |
| `data/fission_random_sampler.py` | migrated | 16 | 7 | 未变 |
| `data/generate_1m_pure_random_pool.py` | migrated | 1 | 2 | 未变 |
| `data/generate_bams_ablation_datasets.py` | migrated | 2 | 5 | 未变 |
| `data/generate_distribution_ablation_twins.py` | migrated | 1 | 4 | 未变 |
| `data/generate_kdpp_curv_gated_dataset.py` | migrated | 3 | 3 | 未变 |
| `data/generate_kdpp_gated_dataset.py` | migrated | 2 | 3 | 未变 |
| `data/generate_motif_topology_dataset.py` | migrated | 2 | 1 | 未变 |
| `data/generate_pure_random_dataset.py` | migrated | 1 | 1 | 未变 |
| `data/gibbs_boltzmann_sampler.py` | migrated | 2 | 2 | 未变 |
| `data/gibbs_sobol_sampler.py` | migrated | 3 | 2 | 未变 |
| `data/mass_dataset_generator.py` | migrated | 15 | 2 | 未变 |
| `data/mass_random_mutant_dist.py` | migrated | 1 | 2 | 未变 |
| `data/mass_random_sampler.py` | migrated | 1 | 1 | 未变 |
| `data/mass_wt_blind_sampler.py` | migrated | 31 | 2 | 未变 |
| `data/mirror_pure_random_ko_interval.py` | migrated | 0 | 1 | 未变 |
| `data/mutant_aware.py` | migrated | 35 | 3 | 未变 |
| `data/normalization.py` | original_module_unchanged | 0 | 0 | 未变 |
| `data/parse_target.py` | original_module_unchanged | 0 | 0 | 未变 |
| `data/preprocessing_contract.py` | original_module_unchanged | 0 | 0 | 未变 |
| `data/reachability_check.py` | original_module_unchanged | 0 | 0 | 未变 |
| `data/real_mutants_only.py` | excluded_obsolete | 0 | 0 | 未变 |
| `data/simulation_guards.py` | original_module_unchanged | 0 | 0 | 未变 |
| `data/yeast_datamodule.py` | consumer_migrated | 0 | 0 | 已变 |
| `data/yeast_graph.py` | consumer_migrated | 0 | 0 | 已变 |
| `data/yeast_relations.py` | original_module_unchanged | 0 | 0 | 未变 |

## 迁移定义逐项对应

仅忽略 docstring、移动的 import、显式 ma 后端注入及 MASS 可选 RNG 接口。source_match 表示该定义结构相同，不表示它依赖的全局对象、调用顺序或整文件默认行为已相同。类按顶层类计，方法包含在类结构比较中。

| 原定义 | 新定义 | 结构结论 |
| --- | --- | --- |
| data/bams_dataset_generator.py:163 `get_mode_polarity` | data/sampling/bams.py:26 | normalized_source_match |
| data/bams_dataset_generator.py:170 `touch_set` | data/sampling/bams.py:33 | normalized_source_match |
| data/bams_dataset_generator.py:177 `resolve_pair_conflicts` | data/sampling/bams.py:40 | normalized_source_match |
| data/bams_dataset_generator.py:205 `fast_eval_division_status` | data/sampling/bams.py:64 | normalized_source_match |
| data/bams_dataset_generator.py:215 `bisect_bifurcation_dosage` | data/sampling/bams.py:74 | normalized_source_match |
| data/bams_dataset_generator.py:256 `get_next_L1_cell` | data/sampling/bams.py:105 | normalized_source_match |
| data/bams_dataset_generator.py:261 `sample_bams_skeleton_and_modes` | data/sampling/bams.py:110 | normalized_source_match |
| data/bams_dataset_generator.py:410 `rescue_collision_candidate` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/bams_dataset_generator.py:453 `sample_bams_mutant_task` | data/sampling/bams.py:242 | normalized_source_match |
| data/bams_dataset_generator.py:486 `execute_simulation_task` | data/sampling/bams.py:267 | normalized_source_match |
| data/bams_dataset_generator.py:514 `generate_bams_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/bams_dataset_generator.py:699 `main` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/fission_random_sampler.py:86 `WT_parameters` | data/systems/fission/parameters.py:3 | normalized_source_match |
| data/fission_random_sampler.py:137 `_Y_switch` | data/systems/fission/ode.py:5 | normalized_source_match |
| data/fission_random_sampler.py:141 `eqns` | data/systems/fission/ode.py:9 | normalized_source_match |
| data/fission_random_sampler.py:197 `_ratio_MPFa_PP` | data/systems/fission/ode.py:64 | normalized_source_match |
| data/fission_random_sampler.py:202 `ev_start` | data/systems/fission/ode.py:69 | normalized_source_match |
| data/fission_random_sampler.py:208 `ev_sfinish` | data/systems/fission/ode.py:77 | normalized_source_match |
| data/fission_random_sampler.py:214 `ev_division` | data/systems/fission/ode.py:85 | normalized_source_match |
| data/fission_random_sampler.py:225 `_default_birth_state` | data/systems/fission/states.py:6 | normalized_source_match |
| data/fission_random_sampler.py:232 `run_event_sim` | data/systems/fission/simulation.py:9 | source_diff_requires_rule_explanation |
| data/fission_random_sampler.py:305 `resample_uniform` | data/systems/fission/simulation.py:82 | normalized_source_match |
| data/fission_random_sampler.py:327 `assign_labels` | data/systems/fission/phenotypes.py:4 | normalized_source_match |
| data/fission_random_sampler.py:428 `next_L1_cell` | data/sampling/fission_random.py:12 | normalized_source_match |
| data/fission_random_sampler.py:433 `_sample_mode` | data/sampling/fission_random.py:17 | normalized_source_match |
| data/fission_random_sampler.py:442 `_strength` | data/systems/fission/distributions.py:5 | normalized_source_match |
| data/fission_random_sampler.py:458 `sample_random_mutant` | data/sampling/fission_random.py:26 | normalized_source_match |
| data/fission_random_sampler.py:476 `param_vector` | data/systems/fission/parameters.py:50 | normalized_source_match |
| data/fission_random_sampler.py:479 `is_duplicate` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/fission_random_sampler.py:488 `derive_wt_birth_state` | data/systems/fission/states.py:13 | normalized_source_match |
| data/fission_random_sampler.py:506 `mutant_birth_state` | data/systems/fission/states.py:32 | source_diff_requires_rule_explanation |
| data/fission_random_sampler.py:525 `simulate_mutant` | data/systems/fission/simulation.py:101 | source_diff_requires_rule_explanation |
| data/fission_random_sampler.py:580 `_sim_worker` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/fission_random_sampler.py:587 `generate_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/fission_random_sampler.py:743 `main` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_1m_pure_random_pool.py:58 `sample_scaling_factor_sobol` | data/systems/budding/distributions/quantiles.py:13 | source_diff_requires_rule_explanation |
| data/generate_1m_pure_random_pool.py:138 `sample_pure_random_candidate` | data/sampling/sobol.py:65 | normalized_source_match |
| data/generate_1m_pure_random_pool.py:215 `generate_1m_pure_random_pool` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_bams_ablation_datasets.py:79 `sample_ts_alpha` | data/sampling/targeted_ablation.py:9 | normalized_source_match |
| data/generate_bams_ablation_datasets.py:98 `apply_targeted_mutation` | data/sampling/targeted_ablation.py:27 | normalized_source_match |
| data/generate_bams_ablation_datasets.py:171 `generate_targeted_pairs` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_bams_ablation_datasets.py:237 `_worker_sim` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_bams_ablation_datasets.py:243 `run_parallel_simulation` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_bams_ablation_datasets.py:256 `pack_and_save_npz` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_bams_ablation_datasets.py:319 `main` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_distribution_ablation_twins.py:47 `sample_scaling_factor_A` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_distribution_ablation_twins.py:66 `apply_gene_mutation_A` | data/sampling/twins.py:7 | source_diff_requires_rule_explanation |
| data/generate_distribution_ablation_twins.py:172 `sample_twin_mutant_pair` | data/sampling/twins.py:15 | normalized_source_match |
| data/generate_distribution_ablation_twins.py:255 `simulate_twin_task` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_distribution_ablation_twins.py:267 `generate_twins` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_kdpp_curv_gated_dataset.py:44 `SignedDirectedGraphKDPP` | data/sampling/kdpp.py:120 | normalized_source_match |
| data/generate_kdpp_curv_gated_dataset.py:214 `sample_signed_kdpp_mutant` | data/sampling/kdpp.py:253 | normalized_source_match |
| data/generate_kdpp_curv_gated_dataset.py:276 `compute_trajectory_curvature` | data/sampling/kdpp.py:299 | normalized_source_match |
| data/generate_kdpp_curv_gated_dataset.py:288 `run_simulation_task` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_kdpp_curv_gated_dataset.py:300 `generate_kdpp_curv_gated_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_kdpp_curv_gated_dataset.py:516 `main` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_kdpp_gated_dataset.py:40 `GraphDiffusionKDPP` | data/sampling/kdpp.py:5 | normalized_source_match |
| data/generate_kdpp_gated_dataset.py:127 `sample_kdpp_mutant` | data/sampling/kdpp.py:75 | normalized_source_match |
| data/generate_kdpp_gated_dataset.py:189 `run_simulation_task` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_kdpp_gated_dataset.py:198 `generate_kdpp_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_kdpp_gated_dataset.py:385 `main` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_motif_topology_dataset.py:105 `MotifTopologySampler` | data/sampling/motif.py:28 | normalized_source_match |
| data/generate_motif_topology_dataset.py:213 `build_perturbed_sample` | data/sampling/motif.py:112 | normalized_source_match |
| data/generate_motif_topology_dataset.py:251 `generate_motif_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/generate_pure_random_dataset.py:54 `sample_pure_random_mutant` | data/sampling/random.py:4 | normalized_source_match |
| data/generate_pure_random_dataset.py:145 `generate_pure_random_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/gibbs_boltzmann_sampler.py:55 `GibbsBoltzmannGraphSampler` | data/sampling/gibbs.py:6 | normalized_source_match |
| data/gibbs_boltzmann_sampler.py:158 `sample_gibbs_mutant` | data/sampling/gibbs.py:84 | normalized_source_match |
| data/gibbs_boltzmann_sampler.py:222 `run_simulation_task` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/gibbs_boltzmann_sampler.py:231 `generate_gibbs_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/gibbs_sobol_sampler.py:42 `GibbsBoltzmannGraphSampler` | data/sampling/gibbs.py:6 | normalized_source_match |
| data/gibbs_sobol_sampler.py:149 `sample_scaling_factor_sobol` | data/systems/budding/distributions/quantiles.py:13 | normalized_source_match |
| data/gibbs_sobol_sampler.py:236 `sample_gibbs_sobol_mutant` | data/sampling/sobol.py:6 | normalized_source_match |
| data/gibbs_sobol_sampler.py:316 `run_simulation_task` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/gibbs_sobol_sampler.py:325 `generate_gibbs_sobol_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/mass_dataset_generator.py:40 `get_default_params` | data/systems/budding/parameters.py:2 | normalized_source_match |
| data/mass_dataset_generator.py:66 `initial_conds` | data/systems/budding/states.py:8 | normalized_source_match |
| data/mass_dataset_generator.py:81 `GK` | data/systems/budding/ode.py:4 | normalized_source_match |
| data/mass_dataset_generator.py:86 `eqns` | data/systems/budding/ode.py:9 | normalized_source_match |
| data/mass_dataset_generator.py:173 `event1` | data/systems/budding/ode.py:95 | normalized_source_match |
| data/mass_dataset_generator.py:175 `event2` | data/systems/budding/ode.py:99 | normalized_source_match |
| data/mass_dataset_generator.py:177 `event3` | data/systems/budding/ode.py:103 | normalized_source_match |
| data/mass_dataset_generator.py:179 `event4` | data/systems/budding/ode.py:107 | normalized_source_match |
| data/mass_dataset_generator.py:183 `analyze_cell_viability_unified` | data/systems/budding/phenotypes.py:4 | normalized_source_match |
| data/mass_dataset_generator.py:206 `analyze_mass_pattern` | data/systems/budding/phenotypes.py:27 | normalized_source_match |
| data/mass_dataset_generator.py:226 `simulate_mutant` | data/systems/budding/simulation.py:47 | source_diff_requires_rule_explanation |
| data/mass_dataset_generator.py:589 `sample_scaling_factor` | data/systems/budding/distributions/mass.py:4 | normalized_source_match |
| data/mass_dataset_generator.py:656 `get_cached_real_params` | data/systems/budding/real_mutants.py:138 | normalized_source_match |
| data/mass_dataset_generator.py:671 `is_duplicate_real_mutant` | data/systems/budding/real_mutants.py:153 | normalized_source_match |
| data/mass_dataset_generator.py:698 `apply_gene_mutation` | data/systems/budding/mutations/mass.py:4 | normalized_source_match |
| data/mass_dataset_generator.py:912 `sample_mass_mutant` | data/sampling/mass_motifs.py:4 | normalized_source_match |
| data/mass_dataset_generator.py:1165 `generate_mass_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/mass_random_mutant_dist.py:65 `sample_mass_random_dist` | data/sampling/random.py:148 | normalized_source_match |
| data/mass_random_mutant_dist.py:152 `_worker_task_executor` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/mass_random_mutant_dist.py:158 `generate_mass_random_dist_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/mass_random_sampler.py:60 `sample_mass_random` | data/sampling/random.py:79 | normalized_source_match |
| data/mass_random_sampler.py:143 `generate_mass_random_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/mass_wt_blind_sampler.py:136 `_tier2_fold` | data/sampling/wt_blind.py:35 | normalized_source_match |
| data/mass_wt_blind_sampler.py:163 `_bfs_dist` | data/sampling/wt_blind.py:62 | normalized_source_match |
| data/mass_wt_blind_sampler.py:175 `graph_dist` | data/sampling/wt_blind.py:76 | normalized_source_match |
| data/mass_wt_blind_sampler.py:186 `_pair_class` | data/sampling/wt_blind.py:84 | normalized_source_match |
| data/mass_wt_blind_sampler.py:270 `touch_set` | data/sampling/wt_blind.py:104 | normalized_source_match |
| data/mass_wt_blind_sampler.py:318 `LogBandSampler` | data/sampling/wt_blind.py:111 | normalized_source_match |
| data/mass_wt_blind_sampler.py:349 `QuotaPicker` | data/sampling/wt_blind.py:141 | normalized_source_match |
| data/mass_wt_blind_sampler.py:369 `qss_g1` | data/sampling/wt_blind.py:159 | normalized_source_match |
| data/mass_wt_blind_sampler.py:413 `qss_M` | data/sampling/wt_blind.py:197 | normalized_source_match |
| data/mass_wt_blind_sampler.py:537 `_critical_volume` | data/sampling/wt_blind.py:279 | normalized_source_match |
| data/mass_wt_blind_sampler.py:562 `compute_D` | data/sampling/wt_blind.py:312 | normalized_source_match |
| data/mass_wt_blind_sampler.py:630 `gal_alpha_cap` | data/sampling/wt_blind.py:345 | normalized_source_match |
| data/mass_wt_blind_sampler.py:660 `_gal_theta_and_invert` | data/sampling/wt_blind.py:355 | normalized_source_match |
| data/mass_wt_blind_sampler.py:718 `apply_gal_inversions` | data/sampling/wt_blind.py:409 | normalized_source_match |
| data/mass_wt_blind_sampler.py:787 `_weighted_gene` | data/sampling/wt_blind.py:459 | normalized_source_match |
| data/mass_wt_blind_sampler.py:796 `_pair_weight` | data/sampling/wt_blind.py:468 | normalized_source_match |
| data/mass_wt_blind_sampler.py:802 `_sample_L2` | data/sampling/wt_blind.py:474 | normalized_source_match |
| data/mass_wt_blind_sampler.py:818 `QuotaPicker_picker_L2` | data/sampling/wt_blind.py:490 | normalized_source_match |
| data/mass_wt_blind_sampler.py:823 `sample_L2_fallback` | data/sampling/wt_blind.py:494 | normalized_source_match |
| data/mass_wt_blind_sampler.py:828 `_walk_cascade` | data/sampling/wt_blind.py:499 | normalized_source_match |
| data/mass_wt_blind_sampler.py:845 `_sample_star` | data/sampling/wt_blind.py:512 | normalized_source_match |
| data/mass_wt_blind_sampler.py:858 `_sample_antag_core` | data/sampling/wt_blind.py:525 | normalized_source_match |
| data/mass_wt_blind_sampler.py:874 `_sample_orthogonal` | data/sampling/wt_blind.py:541 | normalized_source_match |
| data/mass_wt_blind_sampler.py:886 `sample_skeleton` | data/sampling/wt_blind.py:553 | normalized_source_match |
| data/mass_wt_blind_sampler.py:906 `next_L1_cell` | data/sampling/wt_blind.py:572 | normalized_source_match |
| data/mass_wt_blind_sampler.py:923 `_sample_mode` | data/sampling/wt_blind.py:578 | normalized_source_match |
| data/mass_wt_blind_sampler.py:930 `_resolve_conflicts` | data/sampling/wt_blind.py:585 | normalized_source_match |
| data/mass_wt_blind_sampler.py:956 `build_candidate` | data/sampling/wt_blind.py:609 | source_diff_requires_rule_explanation |
| data/mass_wt_blind_sampler.py:1045 `Ledger` | data/sampling/wt_blind.py:678 | normalized_source_match |
| data/mass_wt_blind_sampler.py:1159 `direction_signs` | data/sampling/wt_blind.py:736 | normalized_source_match |
| data/mass_wt_blind_sampler.py:1173 `make_samplers` | data/sampling/wt_blind.py:749 | normalized_source_match |
| data/mass_wt_blind_sampler.py:1182 `generate_wt_blind_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/mass_wt_blind_sampler.py:1551 `_rollback` | data/sampling/wt_blind.py:752 | normalized_source_match |
| data/mirror_pure_random_ko_interval.py:35 `create_mirror_ko_interval_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/mutant_aware.py:96 `get_default_params` | data/systems/budding/parameters.py:2 | normalized_source_match |
| data/mutant_aware.py:122 `initial_conds` | data/systems/budding/states.py:8 | normalized_source_match |
| data/mutant_aware.py:137 `GK` | data/systems/budding/ode.py:4 | normalized_source_match |
| data/mutant_aware.py:142 `eqns` | data/systems/budding/ode.py:9 | normalized_source_match |
| data/mutant_aware.py:229 `event1` | data/systems/budding/ode.py:95 | normalized_source_match |
| data/mutant_aware.py:231 `event2` | data/systems/budding/ode.py:99 | normalized_source_match |
| data/mutant_aware.py:233 `event3` | data/systems/budding/ode.py:103 | normalized_source_match |
| data/mutant_aware.py:235 `event4` | data/systems/budding/ode.py:107 | normalized_source_match |
| data/mutant_aware.py:239 `analyze_cell_viability_unified` | data/systems/budding/phenotypes.py:4 | normalized_source_match |
| data/mutant_aware.py:262 `analyze_mass_pattern` | data/systems/budding/phenotypes.py:27 | normalized_source_match |
| data/mutant_aware.py:276 `simulate_mutant` | data/systems/budding/simulation.py:47 | source_diff_requires_rule_explanation |
| data/mutant_aware.py:642 `sample_scaling_factor` | data/systems/budding/distributions/mutant_aware.py:4 | normalized_source_match |
| data/mutant_aware.py:770 `get_cached_real_params` | data/systems/budding/real_mutants.py:138 | normalized_source_match |
| data/mutant_aware.py:785 `is_duplicate_real_mutant` | data/systems/budding/real_mutants.py:153 | normalized_source_match |
| data/mutant_aware.py:812 `apply_gene_mutation` | data/systems/budding/mutations/mutant_aware.py:4 | normalized_source_match |
| data/mutant_aware.py:1152 `_bfs_dist` | data/systems/budding/topology.py:58 | normalized_source_match |
| data/mutant_aware.py:1164 `graph_dist` | data/systems/budding/topology.py:71 | normalized_source_match |
| data/mutant_aware.py:1188 `_pair_class` | data/systems/budding/topology.py:97 | normalized_source_match |
| data/mutant_aware.py:1344 `touch_set` | data/systems/budding/catalog.py:255 | normalized_source_match |
| data/mutant_aware.py:1362 `QuotaPicker` | data/sampling/common.py:80 | normalized_source_match |
| data/mutant_aware.py:1394 `_sample_random_genes` | data/sampling/mass.py:27 | normalized_source_match |
| data/mutant_aware.py:1400 `_weighted_gene` | data/sampling/mass.py:33 | normalized_source_match |
| data/mutant_aware.py:1409 `_pair_weight` | data/sampling/mass.py:42 | normalized_source_match |
| data/mutant_aware.py:1415 `sample_L2_fallback` | data/sampling/mass.py:48 | normalized_source_match |
| data/mutant_aware.py:1420 `_sample_L2` | data/sampling/mass.py:53 | normalized_source_match |
| data/mutant_aware.py:1438 `_walk_cascade` | data/sampling/mass.py:71 | normalized_source_match |
| data/mutant_aware.py:1455 `_sample_convergent` | data/sampling/mass.py:88 | normalized_source_match |
| data/mutant_aware.py:1488 `_sample_star` | data/sampling/mass.py:121 | normalized_source_match |
| data/mutant_aware.py:1504 `_sample_antag_core` | data/sampling/mass.py:137 | normalized_source_match |
| data/mutant_aware.py:1520 `_sample_orthogonal` | data/sampling/mass.py:153 | normalized_source_match |
| data/mutant_aware.py:1532 `sample_skeleton` | data/sampling/mass.py:165 | normalized_source_match |
| data/mutant_aware.py:1563 `next_L1_cell` | data/sampling/common.py:12 | normalized_source_match |
| data/mutant_aware.py:1568 `_sample_mode` | data/sampling/common.py:17 | normalized_source_match |
| data/mutant_aware.py:1587 `_resolve_conflicts` | data/sampling/common.py:36 | normalized_source_match |
| data/mutant_aware.py:1632 `_assign_convergent_modes` | data/sampling/mass.py:188 | normalized_source_match |
| data/mutant_aware.py:1668 `sample_mass_mutant` | data/sampling/mass.py:223 | normalized_source_match |
| data/mutant_aware.py:1760 `rescue_collision_candidate` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |
| data/mutant_aware.py:1813 `generate_mass_dataset` | 公共 driver / worker / IO，需文字解释 | replaced_driver_or_wrapper_requires_rule_explanation |

## 每个新文件的来源

| 新文件 | 对应旧文件 |
| --- | --- |
| `data/generate.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/generation/__init__.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/generation/contracts.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/generation/derived.py` | `mirror_pure_random_ko_interval.py`、`generate_bams_ablation_datasets.py` |
| `data/generation/io.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/generation/policies.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/generation/runner.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/recipes/__init__.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/sampling/__init__.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/sampling/bams.py` | `bams_dataset_generator.py` |
| `data/sampling/common.py` | `mutant_aware.py` |
| `data/sampling/factory.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/sampling/fission_random.py` | `fission_random_sampler.py` |
| `data/sampling/gibbs.py` | `gibbs_boltzmann_sampler.py`、`gibbs_sobol_sampler.py` |
| `data/sampling/kdpp.py` | `generate_kdpp_gated_dataset.py`、`generate_kdpp_curv_gated_dataset.py` |
| `data/sampling/mass.py` | `mutant_aware.py` |
| `data/sampling/mass_motifs.py` | `mass_dataset_generator.py` |
| `data/sampling/motif.py` | `generate_motif_topology_dataset.py` |
| `data/sampling/random.py` | `generate_pure_random_dataset.py`、`mass_random_sampler.py`、`mass_random_mutant_dist.py` |
| `data/sampling/sobol.py` | `gibbs_sobol_sampler.py`、`generate_1m_pure_random_pool.py` |
| `data/sampling/targeted_ablation.py` | `generate_bams_ablation_datasets.py` |
| `data/sampling/twins.py` | `generate_distribution_ablation_twins.py` |
| `data/sampling/wt_blind.py` | `mass_wt_blind_sampler.py` |
| `data/sampling/wt_pipeline.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/__init__.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/budding/__init__.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/budding/api.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/budding/catalog.py` | `mutant_aware.py`、`mass_dataset_generator.py` |
| `data/systems/budding/distributions/__init__.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/budding/distributions/mass.py` | `mass_dataset_generator.py` |
| `data/systems/budding/distributions/mutant_aware.py` | `mutant_aware.py` |
| `data/systems/budding/distributions/quantiles.py` | `gibbs_sobol_sampler.py`、`generate_1m_pure_random_pool.py` |
| `data/systems/budding/graph.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/budding/mutations/__init__.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/budding/mutations/mass.py` | `mass_dataset_generator.py` |
| `data/systems/budding/mutations/mutant_aware.py` | `mutant_aware.py` |
| `data/systems/budding/ode.py` | `mutant_aware.py`、`mass_dataset_generator.py` |
| `data/systems/budding/parameters.py` | `mutant_aware.py`、`mass_dataset_generator.py` |
| `data/systems/budding/phenotypes.py` | `mutant_aware.py`、`mass_dataset_generator.py` |
| `data/systems/budding/real_mutants.py` | `mutant_aware.py`、`mass_dataset_generator.py` |
| `data/systems/budding/simulation.py` | `mutant_aware.py`、`mass_dataset_generator.py` |
| `data/systems/budding/specification.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/budding/states.py` | `mutant_aware.py`、`mass_dataset_generator.py` |
| `data/systems/budding/topology.py` | `mutant_aware.py`、`generate_motif_topology_dataset.py`、`bams_dataset_generator.py` |
| `data/systems/fission/__init__.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/fission/catalog.py` | `fission_random_sampler.py` |
| `data/systems/fission/constants.py` | `fission_random_sampler.py` |
| `data/systems/fission/distributions.py` | `fission_random_sampler.py` |
| `data/systems/fission/ode.py` | `fission_random_sampler.py` |
| `data/systems/fission/parameters.py` | `fission_random_sampler.py` |
| `data/systems/fission/phenotypes.py` | `fission_random_sampler.py` |
| `data/systems/fission/real_mutants.py` | `fission_random_sampler.py` |
| `data/systems/fission/simulation.py` | `fission_random_sampler.py` |
| `data/systems/fission/specification.py` | 新增公共配置/执行/序列化/系统装配或包初始化；无一对一旧定义 |
| `data/systems/fission/states.py` | `fission_random_sampler.py` |

## 读取方式

`migrated` 是迁移的 17 个生成文件；`excluded_obsolete` 是不迁移的 LHS/real_mutants_only；`consumer_migrated` 是已切换到新接口的辅助模块；`original_module_unchanged` 是原样保留的其他工具模块。

结构计数：`{"excluded_obsolete": 220, "consumer_migrated": 17, "normalized_source_match": 125, "replaced_driver_or_wrapper_requires_rule_explanation": 36, "original_module_unchanged": 42, "source_diff_requires_rule_explanation": 8}`。

完整 JSON 同时记录旧函数默认值、每个 argparse 选项的原默认表达式、规则常量及旧 NPZ 键集合。

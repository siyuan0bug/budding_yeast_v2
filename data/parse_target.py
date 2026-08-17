import sys

DEFAULT_TARGET_FAMILIES = {
    'cdc20_ts': {
        'mutants': [
            '73_cdc20_ts', '78_cdc20_ts_mad2', '79_cdc20_ts_bub2',
            '74_cdc20_clb5', '85_GAL_ESP1_cdc20_ts',
            '71_CLB5_db_pds1_cdc20', '75_cdc20_pds1', '102_net1_cdc20_ts'
        ],
        'num_per_mutant': 500
    },
    'CLB1_clb2': {
        'mutants': [
            '47_clb1_clb2_KO', '48_CLB1_clb2',
            '51_CLB1_clb2_cdh1', '52_CLB1_clb2_pds1', '111_TAB6_1_CLB1_clb2'
        ],
        'num_per_mutant': 500
    },
    'cdh1': {
        'mutants': [
            '41_sic1_cdc6_cdh1', '21_cln1_cln2_cln3_cdh1', '26_cln1_cln2_cln3_apc_ts',
            '42_sic1_cdc6_cdh1_GAL_CDC20', '32_GAL_SIC1_cln1_cln2_cdh1',
            '40_cdh1_cdc6_42_49', '122_APC_A_cdh1_multi_CDC20',
            '106_cdc14_ts_cdh1', '36_sic1_cdh1', '34_cdh1_KO'
        ],
        'num_per_mutant': 500
    },
    'low_lhs_count': {
        'mutants': [
            '88_tem1_ts_multi_CDC15',
            '89_tem1_ts_GAL_CDC15',
            '15_cln3_bck2_multi_CLN2',
            '53_GAL_CLB2_sic1'
        ],
        'num_per_mutant': 300
    }
}

V5_EXPANSION_FAMILIES = {
    'KO_family_fixed': {
        'mutants': ['86_tem1_KO', '92_cdc15_KO', '94_cdc15_ts_multi_TEM1'],
        'num_per_mutant': 500
    },
    'tem1_ts_family': {
        'mutants': ['88_tem1_ts_multi_CDC15', '89_tem1_ts_GAL_CDC15', '90_tem1_net1_ts', '91_tem1_ts_multi_CDC14'],
        'num_per_mutant': 1200
    },
    'cdc14_ts_family': {
        'mutants': ['99_cdc14_ts', '103_cdc14_ts_GAL_SIC1', '104_cdc14_ts_then_GAL_SIC1', '105_cdc14_ts_sic1', '106_cdc14_ts_cdh1', '107_cdc14_ts_GAL_CLN2'],
        'num_per_mutant': 800
    },
    'swi5_cdh1_family': {
        'mutants': ['43_swi5_KO', '44_swi5_GAL_CLB2', '45_swi5_cdh1', '46_swi5_cdh1_GAL_SIC1'],
        'num_per_mutant': 500
    },
    'cln1_cln2_KO_family': {
        'mutants': ['3_cln1_cln2_KO', '4_GAL_CLN2_cln1_cln2_KO', '5_cln1_cln2_sic1_KO', '6_cln1_cln2_cdh1_KO', '7_GAL_CLN2_cln1_cln2_cdh1_KO', '12_cln1_cln2_bck2_KO', '30_GAL_SIC1_cln1_cln2', '31_GAL_SIC1_GAL_CLN2_cln1_cln2', '32_GAL_SIC1_cln1_cln2_cdh1'],
        'num_per_mutant': 400
    },
    'sic1_KO_family': {
        'mutants': ['27_sic1_KO', '28_GAL_SIC1', '29_GAL_SIC1_db'],
        'num_per_mutant': 400
    },
    'GAL_CLB2_cdh1_family': {
        'mutants': ['33_GAL_SIC1_GAL_CLN2_cdh1', '51_CLB1_clb2_cdh1', '54_GAL_CLB2_cdh1'],
        'num_per_mutant': 500
    },
    'cln_triple_extended': {
        'mutants': ['20_cln1_cln2_cln3_sic1', '22_cln1_cln2_cln3_multi_CLB5', '23_cln1_cln2_cln3_GAL_CLB5', '64_cln1_cln2_clb5_clb6', '14_cln3_bck2_GAL_CLN2_cln1_cln2', '17_cln1_cln3_KO', '19_cln1_cln2_cln3_GAL_CLN3', '21_cln1_cln2_cln3_cdh1', '24_cln1_cln2_cln3_multi_BCK2', '25_cln1_cln2_cln3_GAL_CLB2', '26_cln1_cln2_cln3_apc_ts'],
        'num_per_mutant': 500
    },
    'CLB1_clb2_extended': {
        'mutants': ['48_CLB1_clb2', '51_CLB1_clb2_cdh1', '52_CLB1_clb2_pds1', '111_TAB6_1_CLB1_clb2'],
        'num_per_mutant': 1000
    },
    'CLB2_db_enhanced': {
        'mutants': ['55_CLB2_db', '56_CLB2_db_Gal', '60_CLB2_db_clb5', '61.1_CLB2_db_clb5_Gal', '62_GAL_CLB2_db', '57_CLB2_db_multi_SIC1', '58_CLB2_db_GAL_SIC1', '59_CLB2_db_multi_CDC6'],
        'num_per_mutant': 800
    },
    'cdh1_contamination_fix': {
        'mutants': ['116_APC_A_cdh1', '46_swi5_cdh1_GAL_SIC1', '117_APC_A_cdh1_Gal', '6_cln1_cln2_cdh1_KO'],
        'num_per_mutant': 800
    },
    'contamination_fix': {
        'mutants': ['76_cdc20_pds1_clb5', '70_CLB5_db_pds1', '123_APC_A_sic1', '66_GAL_CLB5_sic1', '39_sic1_cdc6_42_49', '69_CLB5_db_sic1', '13_cln3_bck2_KO', '16_cln3_bck2_sic1_KO', '110_TAB6_1_clb5_clb6'],
        'num_per_mutant': 800
    },
    'cdc6_family': {
        'mutants': ['38_cdc6_42_49', '39_sic1_cdc6_42_49', '40_cdh1_cdc6_42_49', '41_sic1_cdc6_cdh1', '42_sic1_cdc6_cdh1_GAL_CDC20'],
        'num_per_mutant': 600
    },
    'CLB2_db_clb5_Raff_family': {
        'mutants': ['61.2_CLB2_db_clb5_Raff'],
        'num_per_mutant': 400
    },
    'APC_A_GAL_CLB2_family': {
        'mutants': ['124_APC_A_GAL_CLB2'],
        'num_per_mutant': 400
    },
    'cdc15_net1_ts_family': {
        'mutants': ['95_cdc15_net1_ts', '109_TAB6_1_cdc15'],
        'num_per_mutant': 400
    },
    'sic1_cdh1_family': {
        'mutants': ['36_sic1_cdh1', '37_sic1_cdh1_GAL_CDC20'],
        'num_per_mutant': 400
    }
}

# dictionary merging logic:
tasks = set()
for fam_name, fam_dict in DEFAULT_TARGET_FAMILIES.items():
    for mut in fam_dict['mutants']:
        for i in range(fam_dict['num_per_mutant']):
            tasks.add(f"{mut}_LHS_T{i:03d}")

for fam_name, fam_dict in V5_EXPANSION_FAMILIES.items():
    for mut in fam_dict['mutants']:
        for i in range(fam_dict['num_per_mutant']):
            tasks.add(f"{mut}_LHS_T{i:03d}")

mutant_counts = {}
for task in tasks:
    mut = task.split('_LHS_T')[0]
    mutant_counts[mut] = mutant_counts.get(mut, 0) + 1

for mut, count in sorted(mutant_counts.items(), key=lambda x: int(x[0].split('_')[0]) if x[0].split('_')[0].replace('.', '').isdigit() else x[0]):
    print(f"{mut}: {count}")

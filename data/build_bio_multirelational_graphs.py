#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEGACY hand-written biochemical priors (historical-model compatibility only).

These six relations are NOT the complete direct dependency graph of the current
mutant_aware.py ODE. Functions remain numerically unchanged for historical
exports and circuit/module metadata. Production relation topology now comes
from data/yeast_graph.py and data/yeast_relations.py, including HANO.
Legacy CLI exports require explicit opt-in and a NEW output file; existing graph
assets cannot be overwritten. Circuit masks and the module/species incidence
are distinct metadata, not duplicate state-state adjacency matrices.
"""

import numpy as np

# 39 Biochemical Variable Definitions
(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5,
 nSIC1P, nC2P, nC5P, nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T,
 nCDC20, nCDH1T, nCDH1, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1,
 nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2) = range(39)

NUM_VARS = 39

def build_relational_graphs():
    # 1. Complex Formation & Dissociation (Reversible mass action)
    A_complex = np.zeros((NUM_VARS, NUM_VARS), dtype=np.float32)
    # Sic1 + Clb2 <=> C2
    A_complex[nSIC1, nC2] = A_complex[nC2, nSIC1] = 1.0
    A_complex[nCLB2, nC2] = A_complex[nC2, nCLB2] = 1.0
    # Sic1 + Clb5 <=> C5
    A_complex[nSIC1, nC5] = A_complex[nC5, nSIC1] = 1.0
    A_complex[nCLB5, nC5] = A_complex[nC5, nCLB5] = 1.0
    # Cdc6 + Clb2 <=> F2
    A_complex[nCDC6, nF2] = A_complex[nF2, nCDC6] = 1.0
    A_complex[nCLB2, nF2] = A_complex[nF2, nCLB2] = 1.0
    # Cdc6 + Clb5 <=> F5
    A_complex[nCDC6, nF5] = A_complex[nF5, nCDC6] = 1.0
    A_complex[nCLB5, nF5] = A_complex[nF5, nCLB5] = 1.0
    # Phosphorylated counterparts
    A_complex[nSIC1P, nC2P] = A_complex[nC2P, nSIC1P] = 1.0
    A_complex[nCLB2, nC2P] = A_complex[nC2P, nCLB2] = 1.0
    A_complex[nSIC1P, nC5P] = A_complex[nC5P, nSIC1P] = 1.0
    A_complex[nCLB5, nC5P] = A_complex[nC5P, nCLB5] = 1.0
    A_complex[nCDC6P, nF2P] = A_complex[nF2P, nCDC6P] = 1.0
    A_complex[nCLB2, nF2P] = A_complex[nF2P, nCLB2] = 1.0
    A_complex[nCDC6P, nF5P] = A_complex[nF5P, nCDC6P] = 1.0
    A_complex[nCLB5, nF5P] = A_complex[nF5P, nCLB5] = 1.0
    # Net1 + Cdc14 <=> RENT
    A_complex[nNET1, nRENT] = A_complex[nRENT, nNET1] = 1.0
    A_complex[nCDC14, nRENT] = A_complex[nRENT, nCDC14] = 1.0
    # Pds1 + Esp1 <=> PE (Esp1 bound)
    A_complex[nPDS1, nESP1] = A_complex[nESP1, nPDS1] = 1.0

    # 2. Kinase Phosphorylation (Cln2, Clb2, Clb5, Cdc15, etc.)
    A_kinase = np.zeros((NUM_VARS, NUM_VARS), dtype=np.float32)
    # Cyclins phosphorylate Sic1, Cdc6, Cdh1, Net1, Swi5
    for c in [nCLN2, nCLB2, nCLB5]:
        A_kinase[c, nSIC1] = 1.0
        A_kinase[c, nSIC1P] = 1.0
        A_kinase[c, nCDC6] = 1.0
        A_kinase[c, nCDC6P] = 1.0
        A_kinase[c, nC2] = 1.0
        A_kinase[c, nC5] = 1.0
        A_kinase[c, nF2] = 1.0
        A_kinase[c, nF5] = 1.0
        A_kinase[c, nCDH1] = 1.0  # Cln2/Clb5/Clb2 inactivate Cdh1
    # Clb2 phosphorylates Swi5 (inactivation) & IEP (activation)
    A_kinase[nCLB2, nSWI5] = 1.0
    A_kinase[nCLB2, nIEP] = 1.0
    # Cdc15 phosphorylates Net1 (MEN pathway, release Cdc14)
    A_kinase[nCDC15, nNET1] = 1.0
    A_kinase[nCDC15, nRENT] = 1.0

    # 3. Phosphatase Dephosphorylation (Cdc14, PPX)
    A_phosphatase = np.zeros((NUM_VARS, NUM_VARS), dtype=np.float32)
    # Cdc14 dephosphorylates Sic1P, Cdc6P, C2P, C5P, F2P, F5P, Cdh1, Swi5
    for sub in [nSIC1P, nSIC1, nCDC6P, nCDC6, nC2P, nC2, nC5P, nC5, nF2P, nF2, nF5P, nF5, nCDH1, nSWI5]:
        A_phosphatase[nCDC14, sub] = 1.0
    # PPX dephosphorylates Net1P (FEAR pathway)
    A_phosphatase[nPPX, nNET1] = 1.0
    A_phosphatase[nPPX, nRENT] = 1.0

    # 4. Transcription Factor Activation (SBF, Mcm1, Swi5)
    A_transcription = np.zeros((NUM_VARS, NUM_VARS), dtype=np.float32)
    # SBF targets: Cln2, Clb5, Pds1
    # Note: SBF is activated by Cln2, Clb5, Mass, inhibited by Clb2
    for act in [nMASS, nCLN2, nCLB5]:
        A_transcription[act, nCLN2] = 1.0
        A_transcription[act, nCLB5] = 1.0
        A_transcription[act, nPDS1] = 1.0
    # Mcm1 targets: Clb2, Swi5, Cdc20
    # Activated by Clb2, Clb5
    for act in [nCLB2, nCLB5]:
        A_transcription[act, nCLB2] = 1.0
        A_transcription[act, nSWI5T] = 1.0
        A_transcription[act, nSWI5] = 1.0
        A_transcription[act, nCDC20T] = 1.0
        A_transcription[act, nCDC20] = 1.0
    # Swi5 targets: Sic1, Cdc6
    A_transcription[nSWI5, nSIC1] = 1.0
    A_transcription[nSWI5, nCDC6] = 1.0

    # 5. APC/C Ubiquitination Proteolysis Degradation (Cdh1, Cdc20)
    A_ubiquitination = np.zeros((NUM_VARS, NUM_VARS), dtype=np.float32)
    # Cdh1 degrades Clb2, Cdc20, Pds1, Swi5
    for deg in [nCLB2, nC2, nC2P, nF2, nF2P, nCDC20T, nCDC20, nPDS1, nSWI5T, nSWI5]:
        A_ubiquitination[nCDH1, deg] = 1.0
    # Cdc20 degrades Clb2, Clb5, Pds1
    for deg in [nCLB2, nC2, nC2P, nF2, nF2P, nCLB5, nC5, nC5P, nF5, nF5P, nPDS1]:
        A_ubiquitination[nCDC20, deg] = 1.0

    # 6. Goldbeter-Koshland & Checkpoint Ultra-sensitive Switch
    A_gk_switch = np.zeros((NUM_VARS, NUM_VARS), dtype=np.float32)
    # Cdh1 switch: Cdc14 (Va) vs Cyclins (Vi)
    A_gk_switch[nCDC14, nCDH1] = 1.0
    for vi in [nCLN2, nCLB2, nCLB5]:
        A_gk_switch[vi, nCDH1] = 1.0
    # SBF switch: Cln2/Clb5/Mass (Va) vs Clb2 (Vi)
    for va in [nMASS, nCLN2, nCLB5]:
        A_gk_switch[va, nCLN2] = 1.0
    A_gk_switch[nCLB2, nCLN2] = 1.0
    # Tem1/Cdc15 switch: lte1 (Va) vs Bub2 (Vi)
    A_gk_switch[nlte1, nTEM1] = 1.0
    A_gk_switch[nBUB2, nTEM1] = 1.0
    A_gk_switch[nTEM1, nCDC15] = 1.0
    # Checkpoint signals: Mad2 (Vi20) inhibits Cdc20 activation
    A_gk_switch[nVi20, nCDC20] = 1.0

    rel_graphs = np.stack([
        A_complex, A_kinase, A_phosphatase,
        A_transcription, A_ubiquitination, A_gk_switch
    ], axis=0) # (6, 39, 39)

    return rel_graphs

def build_circuit_motifs():
    # 4 Functional Circuit Masks (4, 39)
    # Head 1: START / G1-S Transition Circuit
    M_start = np.zeros(NUM_VARS, dtype=np.float32)
    for idx in [nMASS, nCLN2, nCLB5, nSIC1, nCDC6, nC2, nC5, nSIC1P, nCDC6P]:
        M_start[idx] = 1.0

    # Head 2: S/G2/M Replication & Spindle Alignment Checkpoint Circuit
    M_spindle = np.zeros(NUM_VARS, dtype=np.float32)
    for idx in [nCLB5, nCLB2, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nBUB2]:
        M_spindle[idx] = 1.0

    # Head 3: Mitotic Exit Network (MEN) & FEAR Pathway Circuit
    M_men = np.zeros(NUM_VARS, dtype=np.float32)
    for idx in [nCLB2, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15, nPPX, nCDH1T, nCDH1, nSWI5T, nSWI5]:
        M_men[idx] = 1.0

    # Head 4: Cytokinesis, Size Reset & Origin Licensing Reset Circuit
    M_reset = np.zeros(NUM_VARS, dtype=np.float32)
    for idx in [nMASS, nCLN2, nCLB2, nORI, nBUD, nSPN, nlte1]:
        M_reset[idx] = 1.0

    circuit_motifs = np.stack([M_start, M_spindle, M_men, M_reset], axis=0) # (4, 39)
    return circuit_motifs

def build_gene_bipartite_incidence():
    """
    27 Gene Groups (G1 ~ G27) from mass_dataset_generator.py to 39 State Variables
    Shape: (27, 39)
    """
    B_gene = np.zeros((27, NUM_VARS), dtype=np.float32)
    
    # G1: CLN2 -> CLN2
    B_gene[0, nCLN2] = 1.0
    # G2: CLN3 -> MASS, CLN2 (via SBF)
    B_gene[1, nMASS] = B_gene[1, nCLN2] = 1.0
    # G3: CLB2 -> CLB2, C2, F2, C2P, F2P
    for idx in [nCLB2, nC2, nF2, nC2P, nF2P]: B_gene[2, idx] = 1.0
    # G4: CLB5 -> CLB5, C5, F5, C5P, F5P
    for idx in [nCLB5, nC5, nF5, nC5P, nF5P]: B_gene[3, idx] = 1.0
    # G5: SIC1 -> SIC1, C2, C5, SIC1P, C2P, C5P
    for idx in [nSIC1, nC2, nC5, nSIC1P, nC2P, nC5P]: B_gene[4, idx] = 1.0
    # G6: CDC6 -> CDC6, F2, F5, CDC6P, F2P, F5P
    for idx in [nCDC6, nF2, nF5, nCDC6P, nF2P, nF5P]: B_gene[5, idx] = 1.0
    # G7: BCK2 -> MASS, CLN2
    B_gene[6, nMASS] = B_gene[6, nCLN2] = 1.0
    # G8: SWI5 -> SWI5T, SWI5, SIC1, CDC6
    for idx in [nSWI5T, nSWI5, nSIC1, nCDC6]: B_gene[7, idx] = 1.0
    # G9: CDC20 -> CDC20T, CDC20, IEP
    for idx in [nCDC20T, nCDC20, nIEP]: B_gene[8, idx] = 1.0
    # G10: CDH1 -> CDH1T, CDH1
    for idx in [nCDH1T, nCDH1]: B_gene[9, idx] = 1.0
    # G11: CDC14 -> CDC14T, CDC14, RENT
    for idx in [nCDC14T, nCDC14, nRENT]: B_gene[10, idx] = 1.0
    # G12: NET1 -> NET1T, NET1, RENT
    for idx in [nNET1T, nNET1, nRENT]: B_gene[11, idx] = 1.0
    # G13: TEM1 -> TEM1
    B_gene[12, nTEM1] = 1.0
    # G14: CDC15 -> CDC15
    B_gene[13, nCDC15] = 1.0
    # G15: PPX -> PPX
    B_gene[14, nPPX] = 1.0
    # G16: PDS1 -> PDS1, ESP1
    for idx in [nPDS1, nESP1]: B_gene[15, idx] = 1.0
    # G17: ESP1 -> ESP1, PDS1
    for idx in [nESP1, nPDS1]: B_gene[16, idx] = 1.0
    # G18: MAD2 -> Vi20, CDC20
    for idx in [nVi20, nCDC20]: B_gene[17, idx] = 1.0
    # G19: BUB2 -> BUB2, TEM1
    for idx in [nBUB2, nTEM1]: B_gene[18, idx] = 1.0
    # G20: LTE1 -> lte1, TEM1
    for idx in [nlte1, nTEM1]: B_gene[19, idx] = 1.0
    # G21: SBF -> CLN2, CLB5
    for idx in [nCLN2, nCLB5]: B_gene[20, idx] = 1.0
    # G22: MCM1 -> CLB2, SWI5, CDC20
    for idx in [nCLB2, nSWI5, nCDC20]: B_gene[21, idx] = 1.0
    # G23: MDT -> MASS
    B_gene[22, nMASS] = 1.0
    # G24: SCC1 -> SPN, ESP1
    for idx in [nSPN, nESP1]: B_gene[23, idx] = 1.0
    # G25: SWE1 -> CLB2
    B_gene[24, nCLB2] = 1.0
    # G26: MIH1 -> CLB2
    B_gene[25, nCLB2] = 1.0
    # G27: CHECKPOINTS -> ORI, BUD, SPN
    for idx in [nORI, nBUD, nSPN]: B_gene[26, idx] = 1.0

    return B_gene

def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-export', action='store_true',
                        help='Explicitly export the frozen historical priors, NOT the ODE topology')
    parser.add_argument('--output', type=Path, help='A NEW .npz file; existing files are never overwritten')
    args = parser.parse_args(argv)
    if not args.legacy_export or args.output is None:
        parser.error('Legacy exports are frozen by default. For source-derived topology use '
                     'scripts/analysis/dataset_diagnosis/audit_yeast_graph.py; '
                     'historical reproduction requires --legacy-export --output NEW_FILE.npz')
    if args.output.suffix.lower() != '.npz':
        parser.error('--output must be a new .npz file')

    rel_graphs = build_relational_graphs()
    circuit_motifs = build_circuit_motifs()
    gene_bipartite = build_gene_bipartite_incidence()

    # A file handle avoids numpy appending suffixes; exclusive creation prevents
    # accidentally replacing the ignored binary used by historical checkpoints.
    with args.output.open('xb') as stream:
        np.savez_compressed(
            stream,
            rel_graphs=rel_graphs,           # (6, 39, 39)
            circuit_motifs=circuit_motifs,   # (4, 39)
            gene_bipartite=gene_bipartite,   # (27, 39)
            relation_names=[
                "complex", "kinase", "phosphatase",
                "transcription", "ubiquitination", "gk_switch"
            ],
            circuit_names=["START", "SPINDLE", "MEN", "RESET"],
        )
    print(f"Legacy priors exported (not authoritative ODE topology): {args.output.resolve()}")
    print(f"   - Relational Graphs Shape  : {rel_graphs.shape} (Total non-zero edges: {np.count_nonzero(rel_graphs)})")
    print(f"   - Circuit Motifs Shape     : {circuit_motifs.shape}")
    print(f"   - Gene-to-Protein Bipartite: {gene_bipartite.shape}")

if __name__ == "__main__":
    main()

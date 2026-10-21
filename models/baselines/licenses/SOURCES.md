# Source attribution

- GNOT attention and expert blocks: Hao et al., `baselines/GNOT/models/mmgpt.py`. No license file was present in this checkout; no replacement license is asserted.
- LNO: Wang & Wang, `LatentNeuralOperator/LNO-PyTorch/ForwardProblem/module/model.py`, retained vanilla attention and PhCA; MIT text included.
- Transolver: Wu et al., `transolver/Physics_Attention.py` and irregular-mesh residual blocks; MIT text included.
- LaMO: Tiwari et al., `LaMO/model/mamba_2D_v2.py`; Hydra implementation credits Hwang, Lahoti, Puduppully, Dao and Gu (2024). MIT text included. The unfused path uses reference tensor SSD and gated normalization, without compiled Mamba dependencies.
- MWT filter construction: `Tau_BNO-main/models/MWNO/utils.py`, original method Gupta, Xiao & Bogdan; Tau_BNO MIT text included.
- FNO structure: Li et al. / RealPDEBench `model/fno.py`, reduced to one temporal axis. RealPDEBench is CC BY-NC 4.0; task adaptations are documented.
- WNO structure: Tripura & Chakraborty via Tau_BNO `models/WNO`; ptwt backend and joint-state I/O are documented adaptations.
- KoNODE equations: Bai & Ding, `KoNODE/main.py`, rewritten as a batched augmented vector field with condition-dependent initialization. No upstream license file was present; no replacement license is asserted.
- SC-FNO training method: Behroozi, Shen & Kifer, official ODE sensitivity notebooks. Requires reference sensitivity targets, not parameter modulation.

- KAN: Blealtan efficient-kan, MIT, commit `7b6ce1c87f18c8bc90c208f6b494042344216b11`. Core copied unchanged to `_upstream/kan.py`; task input/output wrapper is separate. EfficientKAN_MIT.txt included.
- Transformer and Mamba regressor backbones: Tau_BNO-main/models/OperatorFree/{TransformerBackbone,MambaBackbone,NN,layers}.py; retain RMSNorm/MLP and backbone calculations, remove project imports/BaseModel facade, set declared width/depth/dropout. Tau_BNO MIT text included.
- Mamba-1: Tri Dao and Albert Gu, state-spaces/mamba commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`, Apache-2.0. Retain official initialization and selective_scan_ref with equivalent Torch layout operations; implement the official unfused full-sequence forward without streaming cache or fused inner kernel. Dispatch to official selective_scan_fn on CUDA if installed, otherwise use the retained PyTorch reference. Mamba_APACHE.txt included. Remote source URLs and original SHA256 are in `_upstream/operatorfree_sources.json`.

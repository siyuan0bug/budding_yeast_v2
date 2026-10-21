"""Export the reviewed baseline selection, source hashes and implementation status.

Status is a human source-review conclusion, not inferred numerical equivalence.
Run --require-ready to prevent a pending/legacy entry entering a paper run list.
"""
import argparse
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent

SELECTION = [
    ('MLP','mlp','standard_task_adaptation','models/baselines/mlp.py','Plain concatenated-input MLP.'),
    ('U-Net','unet','standard_task_adaptation','models/baselines/unet.py','Plain temporal encoder/decoder with pooling, transposed convolution and skip concatenation.'),
    ('ResNet','resnet','standard_task_adaptation','models/baselines/resnet.py','Plain residual MLP, no channel attention.'),
    ('Efficient-KAN','kan','standard_task_adaptation','models/baselines/kan.py','Pinned efficient-kan B-spline core, pointwise full-state/parameter/time regression.'),
    ('Transformer','transformer','standard_task_adaptation','models/baselines/transformer.py','Tau_BNO per-token encoder, RMSNorm and sinusoidal positions; full conditions at every time.'),
    ('Mamba-1','mamba','standard_task_adaptation','models/baselines/mamba.py','Tau_BNO regressor with pinned official Mamba-1 initialization and selective scan; reference PyTorch backend without compiled extension.'),
    ('DeepONet','deeponet','standard_task_adaptation','models/baselines/deeponet.py','Branch/trunk with full initial state and parameters.'),
    ('FNO','fno','standard_task_adaptation','models/baselines/fno.py','1D temporal spectral/pointwise operator.'),
    ('WNO','wno','standard_task_adaptation','models/baselines/wno.py','Coarsest low/detail position-dependent kernels; ptwt backend.'),
    ('MWT','mwt','standard_task_adaptation','models/baselines/mwt.py','Polynomial filters and recursive A/B/C hierarchy.'),
    ('GNOT','gnot','standard_task_adaptation','models/baselines/gnot.py','Official heterogeneous normalized attention and geometric experts.'),
    ('Transolver','transolver','standard_task_adaptation','models/baselines/transolver.py','Official slice, slice-token self-attention, deslice.'),
    ('LNO','lno','standard_task_adaptation','models/baselines/lno.py','Official PhCA encoder/decoder and latent vanilla attention.'),
    ('NeuralODE','neural_ode','standard_task_adaptation','models/baselines/neural_ode.py','Learned conditional vector field with actual query times.'),
    ('SC-FNO','sc_fno','standard_task_adaptation','models/baselines/sc_fno.py','Same FNO architecture; training requires sc_sensitivity and explicit reference Jacobians.'),
    ('Bregman-FNO','bregman_fno','standard_task_adaptation','models/baselines/bregman_fno.py','Dual-state postactivation, zero linear updates; configured without optional channel MLP or norm.'),
    ('LaMO','lamo','standard_task_adaptation','models/baselines/lamo.py','Official latent slice/Hydra/deslice; exact unfused SSD tensor backend.'),
    ('KoNODE','konode','standard_task_adaptation','models/baselines/konode.py','Augmented observed state, evolving field weights and Koopman modes.'),
]

SOURCES = [
    'baselines/Tau_BNO-main/models/OperatorFree/KAN.py',
    'baselines/Tau_BNO-main/models/OperatorFree/TransformerBackbone.py',
    'baselines/Tau_BNO-main/models/OperatorFree/MambaBackbone.py',
    'baselines/Tau_BNO-main/models/OperatorFree/NN.py',
    'baselines/Tau_BNO-main/models/OperatorFree/layers.py',
    'baselines/RealPDEBench/realpdebench/model/fno.py',
    'baselines/RealPDEBench/realpdebench/model/MWT_libs/models.py',
    'baselines/Tau_BNO-main/models/WNO/wno.py',
    'baselines/Tau_BNO-main/models/WNO/wavelet_convolution.py',
    'baselines/Tau_BNO-main/models/MWNO/models.py',
    'baselines/Tau_BNO-main/models/MWNO/utils.py',
    'baselines/GNOT/models/mmgpt.py',
    'baselines/transolver/Physics_Attention.py',
    'baselines/LatentNeuralOperator/LNO-PyTorch/ForwardProblem/module/model.py',
    'baselines/bregmano/neuraloperator/neuralop/layers/fno_block.py',
    'baselines/LaMO/model/Latent_SSM_unshared.py',
    'baselines/LaMO/model/mamba_2D_v2.py',
    'baselines/KoNODE/main.py',
    'baselines/SC_Neural_Operators/AMBehroozi-SC_Neural_Operators-3423864/fno_ode /ODEs/ODE1/ode_FNO_IG_EQ_auto_loss.ipynb',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON report path')
    parser.add_argument('--models', nargs='+', help='Filter by current training registry names')
    parser.add_argument('--require-ready', action='store_true',
                        help='Fail if any selected entry is outside the standard adaptation selection')
    args = parser.parse_args()
    entries = [dict(method=method, registry=name, status=status, implementation=path,
                    sha256=digest(PROJECT / path), note=note)
               for method, name, status, path, note in SELECTION]
    if args.models:
        unknown = set(args.models) - {e['registry'] for e in entries}
        if unknown:
            parser.error(f'Names outside the reviewed selection: {sorted(unknown)}')
        entries = [e for e in entries if e['registry'] in args.models]
    report = dict(review_date='2026-10-21',
                  meaning='Standard cores with declared task I/O and numerical backend adaptations; not a training performance claim.',
                  selected=entries,
                  official_sources={path: digest(WORKSPACE / path) for path in SOURCES},
                  pinned_remote_sources=json.loads((PROJECT/'models/baselines/_upstream/operatorfree_sources.json').read_text()),
                  retained_core_sources={str(path.relative_to(PROJECT)):digest(path)
                      for path in sorted((PROJECT/'models/baselines/_upstream').glob('*.py'))})
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end='')
    if args.require_ready and any(e['status'] != 'standard_task_adaptation' for e in entries):
        parser.exit(1, 'Selected entries are not standard task adaptations; see audit report.\n')


if __name__ == '__main__':
    main()

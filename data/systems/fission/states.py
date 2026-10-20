"""Original WT and mutant-specific warmup and fallback protocol."""
import numpy as np
from .constants import N_VAR, iMASS, iRDNA, iPREMPF, iPI, MASS_UNIT
from .parameters import WT_parameters

def _default_birth_state():
    """CellML 占位初值 (模型自洽收敛前的启动状态)。"""
    y = np.zeros(N_VAR)
    y[iMASS], y[iRDNA] = 1.0, 1.0
    y[iPREMPF], y[iPI] = 1.0, 0.2
    return y

def derive_wt_birth_state(verbose=True):
    """预热收敛 WT, 返回其出生态 (最后一次分裂后的状态)。"""
    from .simulation import run_event_sim
    p = WT_parameters()
    p['kS'] = p['kSmin']
    y0 = _default_birth_state()
    segs, info = run_event_sim(y0, p, tf=1200.0)
    if not info['post_div_states']:
        raise RuntimeError('WT warmup failed: no division events. ' + info['err'])
    y_birth = info['post_div_states'][-1].copy()
    cts = np.diff(info['div_times'])[-3:] if len(info['div_times']) >= 2 else []
    if verbose:
        print(f'  [WT warmup] divisions: {len(info["div_times"])}, '
              f'birth mass = {y_birth[iMASS]:.4f} raw '
              f'(= {y_birth[iMASS] / MASS_UNIT:.3f} paper-rel)')
        print(f'  [WT check ] cycle times: '
              f'{[f"{c:.1f}" for c in cts]} min (paper: ~150)')
    return y_birth

def mutant_birth_state(overrides, y_wt_birth, warmup_tf=1000.0, mass_floor=0.05):
    """逐株 warmup: 从严 WT 出生态出发, 把该突变体跑到自身吸引子, 返回其出生态。
    无分裂(G2/M 阻滞)或数值失败/质量塌缩 -> 退回 WT 出生态。"""
    from .simulation import run_event_sim
    p = WT_parameters(); p['kS'] = p['kSmin']
    for k, v in overrides.items():
        if k in p:
            p[k] = v
    y_wtb = np.array(y_wt_birth, float)
    try:
        _, info = run_event_sim(y_wtb.copy(), p, tf=warmup_tf)
    except TimeoutError:
        raise
    except Exception:
        return y_wtb.copy()
    if info['status'] in ('NumericalFailure', 'MaxEvents') or not info['post_div_states']:
        return y_wtb.copy()
    y0 = info['post_div_states'][-1].copy()
    if y0[iMASS] < mass_floor:
        return y_wtb.copy()
    return y0

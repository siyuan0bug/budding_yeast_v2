"""Original event simulation, dense-output resampling, cut and steady-birth label protocol."""
import numpy as np
from .constants import *
from .parameters import WT_parameters, param_vector
from .ode import eqns, EVENTS, _ratio_MPFa_PP
from .states import mutant_birth_state
from .phenotypes import assign_labels

def run_event_sim(y0, p, tf=TF_MIN):
    """事件驱动积分, 逐步记录状态序列, 返回 (seg_list, info)。"""
    from scipy.integrate import solve_ivp
    y0 = np.array(y0, dtype=float).copy()
    y0[iRDNA] = min(y0[iRDNA], 1.0)
    t_cur, n_ev = 0.0, 0
    div_times, div_masses, birth_masses = [], [], []
    rdna_at_div, post_div_states = [], []
    start_times, sfinish_times = [], []
    max_MPF = float(y0[iMPF])
    status, err = 'OK', ''
    segs = []
    while t_cur < tf and n_ev < MAX_EVENTS:
        sol = solve_ivp(eqns, (t_cur, tf), y0, method='BDF', args=(p,),
                        events=EVENTS, dense_output=True, rtol=1e-6, atol=1e-8)
        if not sol.success:
            status, err = 'NumericalFailure', str(sol.message)
            break
        max_MPF = max(max_MPF, float(np.max(sol.sol(np.linspace(t_cur, sol.t[-1], 200))[iMPF])))
        fired = None
        for k, name in ((2, 'DIV'), (0, 'START'), (1, 'SFINISH')):
            if sol.t_events[k].size > 0:
                fired = (k, name, sol.t_events[k][0], sol.y_events[k][0])
                break
        if fired is None:
            segs.append((t_cur, sol.t[-1], sol))
            y0, t_cur = sol.y[:, -1].copy(), sol.t[-1]
            status = 'End'
            break
        k, name, te, ye = fired
        segs.append((t_cur, te, sol))
        ye = np.maximum(np.array(ye, dtype=float), 0.0)
        ye[iRDNA] = min(ye[iRDNA], 1.0)
        if name == 'DIV':
            rdna_at_div.append(float(ye[iRDNA]))   # 分裂时复制完成度 (cut 判据)
            ye[iMASS] /= 2.0
            birth_masses.append(float(ye[iMASS]))
            div_times.append(float(te))
            div_masses.append(float(2.0 * ye[iMASS]))
            # 防止 ratio 恰在阈值处重复触发: 整体微缩 MPF/preMPF 0.05%
            MPFa = ye[iMPF] + p['eps'] * ye[iPREMPF]
            PP = max(1.0 - ye[iPI], 1e-6)
            if PP > 1e-6 and MPFa / PP >= 0.4995:
                ye[iMPF] *= 0.9995
                ye[iPREMPF] *= 0.9995
            post_div_states.append(ye.copy())
        elif name == 'START':
            # 合法 START (G1) 或 S 期虚假穿越均处理: 把 ratio 微推离阈值
            r = _ratio_MPFa_PP(ye, p)
            if ye[iRDNA] >= 1.0:          # G1 中的合法 START
                start_times.append(float(te))
                ye[iRDNA] = 0.0
                p['kS'] = p['kSmax']
            if r <= 0.6502:
                f = 0.6504 / max(r, 1e-12)
                if f <= 1.05:
                    ye[iMPF] *= f
                    ye[iPREMPF] *= f
        else:  # SFINISH
            sfinish_times.append(float(te))
            ye[iRDNA] = 1.0
            p['kS'] = p['kSmin']
        y0, t_cur = ye, float(te) + DT_EPS
        n_ev += 1
    if status == 'OK' and n_ev >= MAX_EVENTS:
        status = 'MaxEvents'
    info = dict(completed_time=float(segs[-1][1]) if segs else 0., div_times=div_times, div_masses=div_masses,
                birth_masses=birth_masses, rdna_at_div=rdna_at_div,
                post_div_states=post_div_states,
                start_times=start_times, sfinish_times=sfinish_times,
                max_MPF=max_MPF, status=status, err=err)
    return segs, info

def resample_uniform(segs, tf=TF_MIN, n_t=N_T):
    """将分段 dense-output 解重采样到均匀网格 (正确处理事件跳变)。"""
    t_grid = np.linspace(0.0, tf, n_t)
    Y = np.zeros((N_VAR, n_t))
    for j, tq in enumerate(t_grid):
        chosen = None
        for (t0, t1, sol) in segs:
            if t0 - 1e-9 <= tq <= t1 + 1e-9:
                chosen = (min(max(tq, t0), t1), sol)
                break
            if t0 > tq:
                chosen = (t0, sol)
                break
        if chosen is None:
            t1, sol = segs[-1][1], segs[-1][2]
            chosen = (t1, sol)
        Y[:, j] = chosen[1].sol(chosen[0])
    return np.maximum(Y, 0.0)

def simulate_mutant(name, overrides, y_wt_birth, level=0, tf=TF_MIN, n_t=N_T):
    """完整仿真一个突变体, 返回紧凑 dict (可 pickle, 供多进程)。"""
    p = WT_parameters()
    p['kS'] = p['kSmin']
    for k, v in overrides.items():
        if k not in p:
            return dict(name=name, level=level, y=None, params_vec=None,
                        label='Invalid Param', pattern=None, n_div=0,
                        birth_rel=float('nan'), status='Invalid',
                        err=f'unknown param {k}')
        p[k] = v
    y0 = mutant_birth_state(overrides, y_wt_birth)
    try:
        segs, info = run_event_sim(y0, p, tf=tf)
    except TimeoutError:
        raise
    except Exception as e:
        return dict(name=name, level=level, y=None, params_vec=param_vector(p),
                    label='Inviable: Numerical Failure', pattern=None, n_div=0,
                    birth_rel=float('nan'), status='NumericalFailure', err=repr(e))
    if info['status'] == 'NumericalFailure':
        return dict(name=name, level=level, y=None, params_vec=param_vector(p),
                    label='Inviable: Numerical Failure', pattern=None, n_div=0,
                    birth_rel=float('nan'), status='NumericalFailure', err=info['err'])
    y = resample_uniform(segs, tf=tf, n_t=n_t)
    if np.max(y[iMASS]) > MASS_LIMIT:
        return dict(name=name, level=level, y=None, params_vec=param_vector(p),
                    label='Discarded_Mass', pattern=None,
                    n_div=len(info['div_times']), birth_rel=float('nan'),
                    status='Discarded_Mass', err='mass explosion')
    # mass 通道归一到论文相对尺度 (WT 出生 = 1.00, 分裂 = 2.00); 其余通道不变
    y[iMASS] = y[iMASS] / MASS_UNIT
    # cut 表型: 未完成复制 (RDNA<0.99) 即发生分裂
    if any(rd < 0.99 for rd in info.get('rdna_at_div', [])):
        return dict(name=name, level=level, y=y, y_init=y0.copy(), completed_time=info['completed_time'], params_vec=param_vector(p),
                    label='Inviable: cut (premature division)', pattern=None,
                    n_div=len(info['div_times']),
                    birth_rel=(info['birth_masses'][-1] / MASS_UNIT
                               if info['birth_masses'] else float('nan')),
                    status=info['status'], err='')
    # Too Small 判据用稳态出生质量: 从最后出生状态再续跑 1050 min 评估稳态
    births_lab = None
    if info['post_div_states']:
        try:
            _, info_ext = run_event_sim(info['post_div_states'][-1], dict(p), tf=1050.0)
            if info_ext['birth_masses']:
                births_lab = info_ext['birth_masses']
        except TimeoutError:
            raise
        except Exception:
            pass
    label, pattern, status = assign_labels(info, y, births_lab=births_lab)
    birth_rel = (births_lab[-1] / MASS_UNIT if births_lab
                 else (info['birth_masses'][-1] / MASS_UNIT if info['birth_masses']
                       else float('nan')))
    return dict(name=name, level=level, y=y, y_init=y0.copy(), completed_time=info['completed_time'], params_vec=param_vector(p),
                label=label, pattern=pattern, n_div=len(info['div_times']),
                birth_rel=birth_rel, status=status, err='')

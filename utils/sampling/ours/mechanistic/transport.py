"""First-order, implicit defect/response transport in physical coordinates.

These are acquisition estimators, not certified bounds on the neural loss.
Finite differences and time discretization have bias. Saltation is only used
at an isolated transverse predicted crossing; unresolved cases are reported.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TransportConfig:
    grid_points: int = 33
    jacobian_step: float = 1e-5
    transverse_tolerance: float = 1e-7
    maximum_events: int = 128

    def validate(self):
        if self.grid_points < 5 or self.maximum_events < 1:
            raise ValueError('Transport needs a positive event cap and at least five grid points')
        if not all(np.isfinite(v) and v > 0 for v in (self.jacobian_step, self.transverse_tolerance)):
            raise ValueError('Transport finite-difference tolerances must be finite and positive')
        return self


def jacobian(system, state, p, t, meta, step):
    return jacobians(system, state[None], p, torch.tensor([t], dtype=state.dtype), meta, step)[0]


def jacobians(system, states, p, times, meta, step):
    """Batch all grid-point perturbations instead of repeated Python RHS calls."""
    k, n = states.shape
    eps = step * torch.maximum(states.abs(), torch.ones_like(states))
    displacement = torch.eye(n, dtype=states.dtype)[None] * eps[:, :, None]
    inputs = torch.cat((states[:, None] + displacement, states[:, None] - displacement), 1).reshape(-1, n)
    values = system.rhs(inputs, p[None].expand(2 * n * k, -1),
                        torch.as_tensor(times).repeat_interleave(2 * n),
                        tuple(meta for _ in range(2 * n * k))).reshape(k, 2 * n, n)
    result = ((values[:, :n] - values[:, n:]) / (2 * eps[:, :, None])).transpose(1, 2)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite continuous Jacobian in mechanism acquisition')
    return result


def jump_matrix(system, state, p, event, t, meta, cfg):
    """Numerical DR and saltation with explicit grazing/branch diagnostics."""
    n = len(state)
    eps = cfg.jacobian_step * torch.maximum(state.abs(), torch.ones_like(state))
    columns = []
    for j in range(n):
        shift = torch.zeros_like(state)
        shift[j] = eps[j]
        columns.append((system.reset(state + shift, p, event, meta)
                        - system.reset(state - shift, p, event, meta)) / (2 * eps[j]))
    dr = torch.stack(columns, 1)
    # Parameters stay fixed: the guard normal is with respect to state only.
    normals = []
    for j in range(n):
        shift = torch.zeros_like(state)
        shift[j] = eps[j]
        v, _ = system.guards(torch.stack((state + shift, state - shift))[None], p[None])
        normals.append((v[0, 0, event] - v[0, 1, event]) / (2 * eps[j]))
    normal = torch.stack(normals)
    fm = system.rhs(state[None], p[None], t, (meta,))[0]
    reset = system.reset(state, p, event, meta)
    fp = system.rhs(reset[None], p[None], t, (meta,))[0]
    denominator = normal @ fm
    threshold = cfg.transverse_tolerance * max(1., float(normal.norm() * fm.norm()))
    if not torch.isfinite(dr).all() or not torch.isfinite(fp).all():
        raise FloatingPointError('Nonfinite event linearization')
    if abs(float(denominator)) <= threshold:
        return dr, False
    return dr + torch.outer(fp - dr @ fm, normal) / denominator, True


def _grid(system, y, p, cfg):
    guards, directions = system.guards(y[None], p[None])
    a, b = guards[0, :-1], guards[0, 1:]
    hit = torch.zeros(len(y) - 1, dtype=torch.bool)
    for j, direction in enumerate(directions):
        hit |= ((a[:, j] >= 0) & (b[:, j] < 0)) if direction < 0 else ((a[:, j] <= 0) & (b[:, j] > 0))
    crossings = torch.where(hit)[0].numpy()
    if len(crossings) > cfg.maximum_events:
        raise ValueError('Too many predicted crossings for bounded mechanism transport')
    # Keep both full-resolution endpoints of every predicted crossing.
    return np.unique(np.r_[np.linspace(0, len(y) - 1, min(cfg.grid_points, len(y))).round().astype(int),
                           crossings, crossings + 1])


def estimate(system, trajectory, p, initial, scale, *, metadata=None, time_grid=None,
             config=TransportConfig(), reference_params=None, reference_initial=None, reference_metadata=None):
    """Return scaled RMS discrepancy plus numerical diagnostics.

    Default: transport the defects of the supplied surrogate trajectory.
    Response mode: supplied trajectory is a *queried training* reference;
    transport the intervention forcing f(Y,p_new)-f(Y,p_reference), including
    the canonical change in initial conditions. Returns the estimated response.
    """
    cfg = config.validate()
    y = torch.as_tensor(np.asarray(trajectory).copy(), dtype=torch.float64).T
    p = torch.as_tensor(np.asarray(p).copy(), dtype=torch.float64)
    initial = torch.as_tensor(system.initials(initial), dtype=torch.float64)
    scale = torch.as_tensor(np.asarray(scale).copy(), dtype=torch.float64)
    if (y.ndim != 2 or y.shape[1] != system.n or len(y) < 5 or scale.shape != (system.n,)
            or not all(torch.isfinite(v).all() for v in (y, p, initial, scale)) or (scale <= 0).any()):
        raise ValueError('Mechanism transport requires finite aligned physical inputs and train scales')
    times = (np.linspace(0., system.horizon, len(y)) if time_grid is None else np.asarray(time_grid))
    if times.shape != (len(y),) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Transport time grid must be finite and strictly increasing')
    meta = {} if metadata is None else metadata
    response = reference_params is not None
    source_meta = meta if not response or reference_metadata is None else reference_metadata
    pa = p if not response else torch.as_tensor(np.asarray(reference_params).copy(), dtype=torch.float64)
    if response and reference_initial is None:
        raise ValueError('Response transport requires the actual reference initial state')
    e = (y[0] - initial if not response else
         initial - torch.as_tensor(system.initials(reference_initial), dtype=torch.float64))
    indices = _grid(system, y, pa if response else p, cfg)
    values = [e.clone()]
    event_count = unresolved = grazing = 0
    eye = torch.eye(system.n, dtype=y.dtype)
    grid_times = torch.as_tensor(times[indices], dtype=y.dtype)
    matrices = jacobians(system, y[indices], p, grid_times, meta, cfg.jacobian_step)
    fb_grid = system.rhs(y[indices], p[None].expand(len(indices), -1), grid_times,
                         tuple(meta for _ in indices))
    fa_grid = (system.rhs(y[indices], pa[None].expand(len(indices), -1), grid_times,
                          tuple(source_meta for _ in indices)) if response else None)
    positions = {int(i): j for j, i in enumerate(indices)}

    def flow(a, b, error, dt, t, grid_index=None):
        j = None if grid_index is None else positions[int(grid_index)]
        matrix = (jacobian(system, b, p, t, meta, cfg.jacobian_step) if j is None else matrices[j])
        fb = system.rhs(b[None], p[None], t, (meta,))[0] if j is None else fb_grid[j]
        if response:
            fa = system.rhs(b[None], pa[None], t, (source_meta,))[0] if j is None else fa_grid[j]
            defect = dt * (fb - fa)
        else:
            defect = b - a - dt * fb
        result = torch.linalg.solve(eye - dt * matrix, error + defect)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite transported error; sample must not be silently zero-scored')
        return result

    for left, right in zip(indices[:-1], indices[1:]):
        dt = float(times[right] - times[left])
        crossing, count = system.event(y[left], y[right], pa if response else p)
        if crossing is None:
            e = flow(y[left], y[right], e, dt, float(times[right]), right)
        else:
            event_count += 1
            unresolved += int(count > 1)
            fraction, event = crossing
            point = y[left] + fraction * (y[right] - y[left])
            t_event = float(times[left] + fraction * dt)
            if fraction > 1e-10:
                e = flow(y[left], point, e, fraction * dt, t_event)
            matrix, transverse = jump_matrix(system, point, pa if response else p, event, t_event, source_meta, cfg)
            grazing += int(not transverse)
            reset = system.reset(point, pa if response else p, event, source_meta)
            e = matrix @ e
            if response:
                e += system.reset(point, p, event, meta) - reset
            if fraction < 1. - 1e-10:
                e = flow(reset, y[right], e, (1. - fraction) * dt, float(times[right]), right)
            elif not response:
                e += y[right] - reset
        values.append(e.clone())
    transported = torch.stack(values)
    energy = (transported / scale).square().mean(1).numpy()
    score = float(np.sqrt(np.trapz(energy, times[indices]) / (times[-1] - times[0])))
    if not np.isfinite(score):
        raise FloatingPointError('Nonfinite transport acquisition score')
    report = dict(estimator='first_order_implicit_transport_v1', mode='response' if response else 'defect',
                  grid_points=len(indices), predicted_events=event_count, unresolved_event_intervals=unresolved,
                  grazing_updates=grazing, certified_error_bound=False,
                  event_time_update='saltation_when_transverse_else_explicitly_reported_reset_jacobian')
    return score, transported.T.numpy(), indices, report

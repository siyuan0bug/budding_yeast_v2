"""Pillar 1: Dual-scale channel-resolved disagreement & joint dynamics-failure embedding."""
import inspect
import numpy as np
import torch

from ...baseline.geometry import gaussian_projection


def _extract_member_outputs(predictor, inputs, *, sketch_dim, eps, return_latent=True):
    """Single batched ensemble pass returning per-channel disagreement and projected latent features."""
    if len(predictor.members) < 2:
        raise ValueError('DMCED requires an ensemble with at least 2 independently trained members')
    n = len(inputs.ids)
    device = predictor.device
    proj = None
    latent_chunks = []
    u_rel_chunks = []
    u_phys_chunks = []

    for start in range(0, n, predictor.batch_size):
        stop = min(start + predictor.batch_size, n)
        x, p = predictor.normalization.encode(inputs.params[start:stop], inputs.initial[start:stop], predictor.steps)
        x_d = x.to(device)
        p_d = p.to(device)
        preds_norm = []
        preds_phys = []
        batch_latent = None

        for m_idx, model in enumerate(predictor.members):
            model.to(device).eval()
            with torch.no_grad():
                backbone = getattr(model, 'model', model)
                if m_idx == 0 and return_latent and 'return_features' in inspect.signature(backbone.forward).parameters:
                    pred_norm, feat = backbone(x_d, p_d, return_features=True)
                    batch_latent = feat.float().flatten(1)
                else:
                    forward = getattr(model, 'forward_ic_time', model)
                    pred_norm = forward(x_d, p_d)
                if not torch.isfinite(pred_norm).all():
                    raise FloatingPointError(f'Nonfinite ensemble prediction before decoding at batch {start}')
                pred_phys = predictor.normalization.decode(pred_norm).float()
                if not torch.isfinite(pred_phys).all():
                    raise FloatingPointError(f'Nonfinite decoded ensemble prediction at batch {start}')
                preds_norm.append(pred_norm.float())
                preds_phys.append(pred_phys)

        stack_norm = torch.stack(preds_norm, dim=0)   # (M, B, V, T)
        stack_phys = torch.stack(preds_phys, dim=0)   # (M, B, V, T)
        mean_norm = stack_norm.mean(dim=0)            # (B, V, T)
        mean_phys = stack_phys.mean(dim=0)            # (B, V, T)

        # Normalized Relative L2 channel disagreement (aligned with training RelativeL2Loss)
        # RMS deviation across ensemble members, L2 across time, divided by smooth mean norm.
        rms_diff_norm = (stack_norm - mean_norm.unsqueeze(0)).square().mean(dim=0).sum(dim=-1).sqrt()
        denom_norm = (mean_norm.square().sum(dim=-1) + eps ** 2).sqrt()
        u_rel = rms_diff_norm / denom_norm            # (B, V)

        # Physical log-amplitude channel disagreement (aligned with physical MAE & zero-channel switches)
        mae_diff_phys = (stack_phys - mean_phys.unsqueeze(0)).abs().mean(dim=(0, 3))
        u_phys = torch.log1p(mae_diff_phys)           # (B, V)

        u_rel_chunks.append(u_rel.cpu().numpy())
        u_phys_chunks.append(u_phys.cpu().numpy())

        if return_latent:
            if batch_latent is None:
                batch_latent = mean_norm.flatten(1)
            if proj is None:
                proj = gaussian_projection(batch_latent.shape[1], int(sketch_dim), predictor.sketch_seed).to(device)
            latent_chunks.append((batch_latent @ proj).cpu().numpy())

    u_rel_all = np.concatenate(u_rel_chunks, axis=0).astype(np.float32)
    u_phys_all = np.concatenate(u_phys_chunks, axis=0).astype(np.float32)
    latent_all = np.concatenate(latent_chunks, axis=0).astype(np.float32) if return_latent else None
    return u_rel_all, u_phys_all, latent_all


def dual_scale_risk(u_rel, u_phys, *, cap_quantile=0.995):
    """Scale-invariant dual epistemic risk w(x) combining relative-norm and log-physical disagreement."""
    r_rel = u_rel.mean(axis=1).astype(np.float64)
    r_phys = u_phys.mean(axis=1).astype(np.float64)
    med_rel = max(float(np.median(r_rel)), 1e-8)
    med_phys = max(float(np.median(r_phys)), 1e-8)
    w = np.sqrt(np.maximum(r_rel / med_rel, 0.0) * np.maximum(r_phys / med_phys, 0.0))
    if len(w) > 10 and 0.0 < cap_quantile < 1.0:
        cap = float(np.quantile(w, cap_quantile))
        if np.isfinite(cap) and cap > 0.0:
            w = np.minimum(w, cap)
    w = np.maximum(w, 1e-6)
    return w, r_rel, r_phys


def _unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return matrix / norms


def build_joint_embeddings(pool_u_rel, pool_u_phys, pool_latent,
                           train_u_rel, train_u_phys, train_latent, *, gamma=0.5):
    """Joint Dynamics-Failure-Mode embedding phi_tilde(x) = [unit(phi_dyn), gamma * unit(u_sig)]."""
    pool_sig_raw = np.concatenate([pool_u_rel, pool_u_phys], axis=1)
    train_sig_raw = np.concatenate([train_u_rel, train_u_phys], axis=1)
    scale = np.maximum(pool_sig_raw.std(axis=0, keepdims=True), 1e-6)
    pool_sig = _unit_rows(pool_sig_raw / scale)
    train_sig = _unit_rows(train_sig_raw / scale)

    pool_dyn = _unit_rows(pool_latent)
    train_dyn = _unit_rows(train_latent)

    pool_embed = np.concatenate([pool_dyn, float(gamma) * pool_sig], axis=1).astype(np.float32)
    train_embed = np.concatenate([train_dyn, float(gamma) * train_sig], axis=1).astype(np.float32)
    return pool_embed, train_embed


def compute_pool_and_train_views(predictor, pool_inputs, train_inputs=None, *,
                                 sketch_dim=512, eps=0.01, gamma=0.5, return_latent=True):
    """Compute dual-scale epistemic risk and (optionally) joint embeddings for pool and train anchors."""
    p_rel, p_phys, p_lat = _extract_member_outputs(
        predictor, pool_inputs, sketch_dim=sketch_dim, eps=eps, return_latent=return_latent
    )
    risk, r_rel, r_phys = dual_scale_risk(p_rel, p_phys)
    result = dict(
        risk=risk,
        r_rel=r_rel,
        r_phys=r_phys,
        u_rel=p_rel,
        u_phys=p_phys,
    )
    if return_latent and train_inputs is not None and len(train_inputs.ids) > 0:
        t_rel, t_phys, t_lat = _extract_member_outputs(
            predictor, train_inputs, sketch_dim=sketch_dim, eps=eps, return_latent=True
        )
        pool_embed, train_embed = build_joint_embeddings(
            p_rel, p_phys, p_lat, t_rel, t_phys, t_lat, gamma=gamma
        )
        result['pool_embed'] = pool_embed
        result['train_embed'] = train_embed
    elif return_latent:
        sig_raw = np.concatenate([p_rel, p_phys], axis=1)
        scale = np.maximum(sig_raw.std(axis=0, keepdims=True), 1e-6)
        result['pool_embed'] = np.concatenate(
            [_unit_rows(p_lat), float(gamma) * _unit_rows(sig_raw / scale)], axis=1
        ).astype(np.float32)
        result['train_embed'] = np.empty((0, result['pool_embed'].shape[1]), dtype=np.float32)
    return result

"""Pillar 1: Uncapped Task-Aligned Epistemic Relative Variance & Unit-Sphere Joint Embedding."""
import inspect
import numpy as np
import torch

from ...baseline.geometry import gaussian_projection


def _unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return matrix / norms


def compute_epistemic_relative_views(predictor, pool_inputs, train_inputs=None, *,
                                     sketch_dim=512, eps=0.01, gamma=0.5, return_latent=True):
    """Single batched ensemble pass computing uncapped squared Relative L2 variance and joint embeddings."""
    if len(predictor.members) < 2:
        raise ValueError('Switch-D2 requires an ensemble with at least 2 independently trained members')

    proj = None

    def _run(inputs, need_latent):
        nonlocal proj
        n = len(inputs.ids)
        device = predictor.device
        e_rel_chunks = []
        e_mse_chunks = []
        lat_chunks = []

        for start in range(0, n, predictor.batch_size):
            stop = min(start + predictor.batch_size, n)
            x, p = predictor.normalization.encode(inputs.params[start:stop], inputs.initial[start:stop], predictor.steps)
            x_d, p_d = x.to(device), p.to(device)
            preds_norm = []
            batch_latent = None

            for m_idx, model in enumerate(predictor.members):
                model.to(device).eval()
                with torch.no_grad():
                    backbone = getattr(model, 'model', model)
                    if m_idx == 0 and need_latent and 'return_features' in inspect.signature(backbone.forward).parameters:
                        pred_norm, feat = backbone(x_d, p_d, return_features=True)
                        batch_latent = feat.float().flatten(1)
                    else:
                        forward = getattr(model, 'forward_ic_time', model)
                        pred_norm = forward(x_d, p_d)
                    if not torch.isfinite(pred_norm).all():
                        raise FloatingPointError(f'Nonfinite ensemble prediction at batch {start}')
                    preds_norm.append(pred_norm.float())

            stack = torch.stack(preds_norm, dim=0)        # (M, B, V, T)
            mean = stack.mean(dim=0)                      # (B, V, T)
            var_t = (stack - mean.unsqueeze(0)).square().mean(dim=0)  # (B, V, T)

            # Exact squared Relative L2 ensemble variance per channel (uncapped!)
            e_rel_ch = var_t.sum(dim=-1) / (mean.square().sum(dim=-1) + eps ** 2)   # (B, V)
            e_mse_ch = var_t.mean(dim=-1)                                           # (B, V)

            e_rel_chunks.append(e_rel_ch.cpu().numpy())
            e_mse_chunks.append(e_mse_ch.cpu().numpy())

            if need_latent:
                if batch_latent is None:
                    batch_latent = mean.flatten(1)
                if proj is None:
                    proj = gaussian_projection(batch_latent.shape[1], int(sketch_dim), predictor.sketch_seed).to(device)
                lat_chunks.append((batch_latent @ proj).cpu().numpy())

        e_rel_all = np.concatenate(e_rel_chunks, axis=0).astype(np.float64)
        e_mse_all = np.concatenate(e_mse_chunks, axis=0).astype(np.float64)
        lat_all = np.concatenate(lat_chunks, axis=0).astype(np.float32) if need_latent else None
        return e_rel_all, e_mse_all, lat_all

    p_rel, p_mse, p_lat = _run(pool_inputs, return_latent)
    # Task-aligned epistemic relative variance (strictly positive, uncapped quadratic energy)
    energy = np.maximum(p_rel.mean(axis=1), 1e-12)

    result = dict(
        energy=energy,
        e_rel_ch=p_rel,
        e_mse_ch=p_mse,
    )

    if return_latent:
        # Log-compressed channel failure signature on the unit sphere (prevents single-channel norm hijack in distance)
        med_ch = np.maximum(np.median(p_rel, axis=0, keepdims=True), 1e-8)
        p_sig = _unit_rows(np.log1p(p_rel / med_ch).astype(np.float32))
        p_dyn = _unit_rows(p_lat)
        result['pool_embed'] = np.concatenate([p_dyn, float(gamma) * p_sig], axis=1).astype(np.float32)

        if train_inputs is not None and len(train_inputs.ids) > 0:
            t_rel, _, t_lat = _run(train_inputs, True)
            t_sig = _unit_rows(np.log1p(t_rel / med_ch).astype(np.float32))
            t_dyn = _unit_rows(t_lat)
            result['train_embed'] = np.concatenate([t_dyn, float(gamma) * t_sig], axis=1).astype(np.float32)
        else:
            result['train_embed'] = np.empty((0, result['pool_embed'].shape[1]), dtype=np.float32)

    return result

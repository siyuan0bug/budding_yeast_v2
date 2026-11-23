"""Stochastic greedy conditional facility coverage and compositional transfer.

Research prototypes: neither function claims an original theoretical guarantee.
The first has a declared nonnegative facility objective on sampled landmarks;
the second is a cross-background adaptation of gradient matching, not a claim
that validation-gradient selection itself is novel.
"""
import math
import numpy as np
import torch


def conditional_vectors(trajectory, owners, effects):
    """Keep joint context/effect coordinates; no named phenotype classes."""
    context = np.asarray(trajectory)[owners]
    context = context/np.maximum(context.std(0, keepdims=True), .1)
    effects = effects/np.maximum(effects.std(0, keepdims=True), .1)
    return np.concatenate((context, effects, context*effects), 1).astype(np.float32)


def facility_select(features, owners, tokens, count, initial, *, seed=42,
                    landmarks=32768, device='cpu', conditional=True):
    """Maximize weighted landmark similarity covered by selected sample IDs.

    Conditional landmarks are balanced across action types and only covered by
    occurrences of the SAME action in a candidate. Within an action, context
    and effect distances both matter. Several occurrences can belong to one
    sample and a query covers them together. Initial samples are pre-covered.
    Ordinary trajectory control uses the same objective/solver with one token.
    """
    features = np.asarray(features, dtype=np.float32)
    owners = np.asarray(owners, dtype=np.int64)
    initial = np.asarray(initial, dtype=np.int64)
    tokens = np.asarray(tokens, dtype=str)
    n = int(owners.max())+1
    if (len(features) != len(owners) or len(tokens) != len(owners) or features.ndim != 2
            or not np.isfinite(features).all() or count < 1
            or len(set(initial)) != len(initial) or not set(initial) <= set(range(n))
            or count+len(initial) > n or set(owners) != set(range(n))):
        raise ValueError('Invalid finite coverage universe/budget')
    rng = np.random.default_rng(seed)
    types = sorted(set(tokens))
    if not conditional:
        tokens = np.full(len(tokens), 'trajectory'); types = ['trajectory']
    per_type = max(1, landmarks//len(types))
    centers, weights = [], []
    for token in types:
        positions = np.flatnonzero(tokens == token)
        chosen = rng.choice(positions, min(per_type, len(positions)), replace=False)
        centers.extend(map(int, chosen))
        weights.extend([1/(len(types)*len(chosen))]*len(chosen))
    centers = np.asarray(centers)
    uncovered_landmark_inputs = set(map(int, owners[centers]))-set(map(int, initial))
    if len(uncovered_landmark_inputs) <= count and n-len(initial) > len(uncovered_landmark_inputs):
        raise ValueError('Too few landmark inputs relative to selection budget; coverage could saturate by selecting all landmarks')
    x = torch.as_tensor(features, device=device)
    reference = x[centers]
    # Positive bandwidth is determined on training-side features only.
    pairs = rng.integers(len(features), size=(2048, 2))
    distance = np.square(features[pairs[:, 0]]-features[pairs[:, 1]]).sum(1)
    bandwidth = max(float(np.median(distance[distance > 0])) if (distance > 0).any() else 1., 1e-6)
    affinity_bytes = n*len(centers)*4
    if str(device).startswith('cuda'):
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        if free < affinity_bytes+2*2**30:
            raise MemoryError(f'Conditional affinity needs {affinity_bytes/2**30:.2f}GiB plus 2GiB headroom; wait for more GPU memory')
    similarity = torch.zeros((n, len(centers)), device=device, dtype=torch.float32)
    for start in range(0, len(features), 1024):
        stop = min(start+1024, len(features))
        squared = torch.cdist(x[start:stop], reference).square()
        value = torch.exp(-squared/bandwidth)
        same = torch.as_tensor(tokens[start:stop, None] == tokens[centers][None], device=device)
        value *= same
        indices = torch.as_tensor(owners[start:stop], device=device)[:, None].expand(-1, len(centers))
        similarity.scatter_reduce_(0, indices, value, reduce='amax', include_self=True)
    del x, reference
    weight = torch.as_tensor(weights, device=device)
    covered = similarity[initial].amax(0) if len(initial) else torch.zeros(len(centers), device=device)
    before = float(covered @ weight)
    available = np.ones(n, bool); available[initial] = False
    selected = []
    # Stochastic greedy, epsilon=.05; exact gains within each sampled choice set.
    sample_size = max(1, math.ceil((n-len(initial))/count*math.log(20)))
    for _ in range(count):
        choices = np.flatnonzero(available)
        if len(choices) > sample_size:
            choices = rng.choice(choices, sample_size, replace=False)
        gain = ((similarity[choices]-covered).clamp_min(0)*weight).sum(1)
        index = int(choices[int(gain.argmax())])
        available[index] = False; selected.append(index)
        covered = torch.maximum(covered, similarity[index])
    return np.asarray(selected, dtype=np.int64), dict(
        objective='mean per-action conditional facility coverage' if conditional else 'trajectory facility coverage',
        landmark_count=len(centers), action_types=len(types), bandwidth=bandwidth,
        landmark_unique_inputs=len(set(map(int, owners[centers]))), affinity_gib=affinity_bytes/2**30,
        initial_objective=before, final_objective=float(covered @ weight),
        stochastic_greedy_epsilon=.05, sample_size=sample_size,
        label_access='all supplied training-side trajectories; not label-limited active learning')


def transfer_select(gradients, candidate_actions, feedback_gradients, feedback_actions,
                    feedback_loss, count, *, seed=42, device='cpu'):
    """Match loss-weighted feedback gradients across different full contexts.

    The caller must reserve complete feedback contexts OUTSIDE candidates.
    One shared action defines a transfer episode. Positive first-order support
    is balanced across actions; a residual target reduces repeated directions.
    Gradients are explicitly head-gradient sketches, not exact full-model gains.
    """
    g = np.asarray(gradients, dtype=np.float32)
    f = np.asarray(feedback_gradients, dtype=np.float32)
    loss = np.asarray(feedback_loss, dtype=np.float32)
    if (g.ndim != 2 or f.ndim != 2 or g.shape[1] != f.shape[1] or not np.isfinite(g).all()
            or not np.isfinite(f).all() or not np.isfinite(loss).all() or not 0 < count <= len(g)):
        raise ValueError('Malformed transfer-gradient bank')
    actions = sorted(set().union(*map(set, candidate_actions)) & set().union(*map(set, feedback_actions)))
    if not actions:
        raise ValueError('No shared interventions in disjoint background episodes')
    targets, risks = [], []
    mask = np.zeros((len(g), len(actions)), np.float32)
    for j, action in enumerate(actions):
        rows = [i for i, tokens in enumerate(feedback_actions) if action in tokens]
        targets.append(f[rows].mean(0)); risks.append(float(loss[rows].mean()))
        mask[:, j] = [action in tokens for tokens in candidate_actions]
    target = torch.tensor(np.asarray(targets), device=device)
    target /= target.norm(dim=1, keepdim=True).clamp_min(1e-12)
    g = torch.as_tensor(g, device=device)
    g /= g.norm(dim=1, keepdim=True).clamp_min(1e-12)
    mask = torch.as_tensor(mask, device=device)
    risk = torch.tensor(risks, device=device)
    risk /= risk.sum().clamp_min(1e-12)
    residual = target.clone()
    rng = np.random.default_rng(seed)
    available = np.ones(len(g), bool); selected = []
    # Blocked residual matching; full rescoring every 64 selected samples.
    for start in range(0, count, 64):
        score = ((g @ residual.T)*mask*risk).sum(1)
        score[torch.as_tensor(~available, device=device)] = -torch.inf
        # Tiny seeded tie-break only, independent of target/test identities.
        score += torch.tensor(rng.uniform(0, 1e-8, len(g)), device=device)
        indices = torch.topk(score, min(64, count-start)).indices
        for i in indices.cpu().tolist():
            selected.append(i); available[i] = False
        update = mask[indices].T @ g[indices]
        residual -= update/(count/max(len(actions), 1))
    return np.asarray(selected), dict(objective='cross-background action-conditioned gradient residual matching',
        action_types=len(actions), feedback_rows=len(f), gradient_dimension=g.shape[1],
        approximation='frozen initial model output-head gradients; first order, blocked selection',
        native_prior_work='gradient matching/validation subset selection; task-specific research adaptation')

"""Exact intervention rectangles and unique-endpoint marginal query accounting."""
from collections import defaultdict
from itertools import combinations
import heapq

import numpy as np


def pairs(meta):
    return tuple(sorted((str(g), str(m)) for g, m in meta.get('pairs', meta.get('genes', ()))))


def rectangles(inputs, system, pool_size, *, maximum=20000):
    """Find existing exact factorial controls, never interpolate physical vectors.

    Initial states may change under a legal intervention. All other parameter
    coordinates, module/mode context and control values must match exactly.
    Overlapping parameter bindings are excluded, rather than treated as two axes.
    """
    bindings = system.module_parameters()
    positions = {name: j for j, name in enumerate(system.param_names)}
    groups = defaultdict(dict)
    for i, (p, meta) in enumerate(zip(inputs.params, inputs.metadata)):
        tokens = pairs(meta)
        modules = sorted({g for g, _ in tokens if g in bindings})
        for a, b in combinations(modules, 2):
            ia = [positions[k] for k in bindings[a] if k in positions]
            ib = [positions[k] for k in bindings[b] if k in positions]
            if not ia or not ib or set(ia) & set(ib):
                continue
            other = np.delete(p, sorted(set(ia + ib))).astype('<f8').tobytes()
            # Non-exported event switches must also agree.
            overrides = meta.get('_effective_overrides', meta.get('original_overrides', {}))
            flags = tuple(bool(overrides.get(k, False))
                          for k in ('is_sbf_ca', 'is_mutant_104'))
            initials = tuple(sorted((k, float(v)) for k, v in overrides.items()
                                    if k.startswith('init_') and k not in set(bindings[a]) | set(bindings[b])))
            key = tokens, a, b, other, flags, initials
            point = p[ia].astype('<f8').tobytes(), p[ib].astype('<f8').tobytes()
            previous = groups[key].get(point)
            if previous is None or (previous < pool_size <= i):
                groups[key][point] = i  # an already queried corner costs no query
    result = []
    for points in groups.values():
        columns = defaultdict(dict)
        for (x, y), i in points.items():
            columns[x][y] = i
        # Bounded enumeration; singleton continuous coordinates cannot form a face.
        columns = {x: rows for x, rows in columns.items() if len(rows) >= 2}
        for a, b in combinations(sorted(columns), 2):
            common = sorted(set(columns[a]) & set(columns[b]))
            for y0, y1 in combinations(common, 2):
                face = (columns[a][y0], columns[a][y1], columns[b][y0], columns[b][y1])
                if any(i < pool_size for i in face):
                    result.append(face)
                if len(result) >= maximum:
                    return np.asarray(result, dtype=np.int64)
    return np.asarray(result, dtype=np.int64).reshape(-1, 4)


def choose_faces(faces, error_features, pool_size, budget, point_scores, seed):
    """Expected interaction-error reduction per NEW endpoint, not |true epistasis|.

    Features are mechanism-estimated errors of unqueried points, zero for queried
    anchors. Selected endpoints stop receiving repeated credit. Incomplete budget
    slots use explicitly reported point-error acquisition, never pretend faces.
    """
    rng = np.random.default_rng(seed)
    adjacency = defaultdict(list)
    for j, face in enumerate(faces):
        for node in face:
            if node < pool_size:
                adjacency[int(node)].append(j)
    selected, chosen, records = [], set(), []
    heap, tickets = [], np.zeros(len(faces), dtype=np.int64)
    signs = np.array([1., -1., -1., 1.])

    def utility(j):
        face = faces[j]
        new = [int(i) for i in face if i < pool_size and i not in chosen]
        if not new:
            return 0., new
        remaining = np.array([i < pool_size and i not in chosen for i in face])
        effect = ((signs * remaining)[:, None] * error_features[face]).sum(0)
        return float(np.sqrt(np.mean(effect ** 2))) / len(new), new

    def push(j):
        value, _ = utility(j)
        tickets[j] += 1
        if value > 0:
            heapq.heappush(heap, (-value, j, int(tickets[j])))

    for j in range(len(faces)):
        push(j)
    while heap and len(selected) < budget:
        _, j, version = heapq.heappop(heap)
        if version != tickets[j]:
            continue
        value, new = utility(j)
        if not new or len(selected) + len(new) > budget:
            continue
        chosen.update(new)
        selected.extend(new)
        records.append(dict(corners=faces[j].tolist(), new_endpoints=new, score=value))
        for affected in {f for node in new for f in adjacency[node]}:
            push(affected)
    face_endpoints = len(selected)
    order = rng.permutation(pool_size)
    order = order[np.argsort(-np.asarray(point_scores)[order], kind='stable')]
    for i in order:
        if len(selected) == budget:
            break
        if int(i) not in chosen:
            selected.append(int(i))
            chosen.add(int(i))
    return np.asarray(selected), dict(exact_faces=len(faces), completed_faces=records,
        face_query_endpoints=face_endpoints, point_error_fill=budget - face_endpoints,
        interaction_target='mechanism-estimated prediction error, not a zero-interaction prior')

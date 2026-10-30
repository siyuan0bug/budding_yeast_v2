"""Explicit mutation descriptions shared by static datasets and AL snapshots."""
import json
import re

import numpy as np

SAMPLE_METADATA_VERSION = 'sample_provenance_v1'


def mutation_metadata(names, metadata_json, is_real, system_id):
    """Keep real benchmark genotypes verbatim, without inventing synthetic alleles.

    Structured pairs describe the final synthetic candidate after rescue. A real
    benchmark can have biology that has no counterpart in the synthetic catalog.
    """
    if len(names) != len(metadata_json) or len(names) != len(is_real):
        raise ValueError('Mutation metadata must align with sample rows')
    descriptions, pairs_json, available = [], [], []
    for name, encoded, real in zip(names, metadata_json, is_real):
        meta = json.loads(str(encoded))
        description = str(meta.get('description') or name)
        pairs = meta.get('pairs', [])
        if not pairs and not real:
            if system_id == 'budding':
                from .systems.budding.samplers.rescue import parse_pairs
                pairs = parse_pairs(description)
            else:
                pairs = re.findall(r'([A-Za-z0-9_]+):([^+\s]+)', description)
        descriptions.append(description)
        pairs_json.append(json.dumps(pairs, ensure_ascii=False))
        available.append(bool(pairs))
    return dict(sample_metadata_version=np.array(SAMPLE_METADATA_VERSION),
                mutation_descriptions=np.asarray(descriptions, dtype=str),
                mutation_pairs_json=np.asarray(pairs_json, dtype=str),
                mutation_pairs_available=np.asarray(available, dtype=bool))

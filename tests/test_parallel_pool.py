"""Parallel input proposals preserve profiles, global uniqueness and fixed-seed order."""
from collections import Counter
from dataclasses import replace
import json

import numpy as np
import pytest

from budding_yeast_v2.data.systems.contracts import GenerationConfig
from budding_yeast_v2.data.systems.parallel_pool import prepare_parallel_pool, RNG_CONTRACT
from budding_yeast_v2.data.systems.registry import get_recipe, resolve_config, get_adapter


@pytest.mark.parametrize('recipe_name',['budding_pool','fission_random'])
def test_parallel_pool_repeatable_across_worker_counts_and_globally_unique(recipe_name):
    config=resolve_config(GenerationConfig(recipe_name,n_samples=520,seed=47,workers=1,parameter_only=True,include_real=False))
    recipe=get_recipe(recipe_name)
    outputs=[]
    for workers in (1,2):
        report=Counter()
        rows=prepare_parallel_pool(recipe,replace(config,workers=workers),report)
        from budding_yeast_v2.data.systems.budding.al_pool import POOL_CONTRACT
        assert report['rng_contract']==(POOL_CONTRACT if recipe_name=='budding_pool' else RNG_CONTRACT)
        assert len(rows)==520
        assert all(row.metadata['module_count']==row.level for row in rows)
        outputs.append([(row.name,json.dumps(row.overrides,sort_keys=True)) for row in rows])
        adapter=get_adapter(recipe.system)
        if recipe.system=='budding':
            from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
            from budding_yeast_v2.data.systems.budding.real_mutants import is_duplicate_real_mutant
            signatures=[]
            for candidate in rows:
                assert not is_duplicate_real_mutant(candidate.overrides,BuddingSystem().defaults())
                result=adapter.parameter_result(candidate,None)
                signatures.append(np.r_[result.params,result.y_init].tobytes())
            assert len(set(signatures))==520
        else:
            from budding_yeast_v2.data.systems.fission.parameters import param_vector,WT_parameters
            index=adapter.references()
            for row in rows:
                vector=param_vector({**WT_parameters(),**row.overrides})
                assert not index.collides(vector)
                index.add(vector)
    assert outputs[0]==outputs[1]

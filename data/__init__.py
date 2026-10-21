"""
budding_yeast_v2.data module
============================
提供数据生成、加载与生化先验常数。
公共生物接口来自 systems.budding；历史生成器保留供显式历史对照。
"""

from .systems.compat import install_legacy_imports as _install_legacy_imports

_install_legacy_imports()

from .systems.budding.api import (
    get_default_params,
    mutant_rules,
    eqns,
    initial_conds,
    PARAM_KEYS,
)
from .yeast_datamodule import YeastDataModule
from .dataset_utils import load_yeast_dataset_universal, stratified_split_subset

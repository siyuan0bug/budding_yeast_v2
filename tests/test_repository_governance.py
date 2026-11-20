"""Repository boundaries are checked statically; no retired implementation executes."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('repository_audit',ROOT/'scripts/maintenance/audit_repository.py')
audit_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit_module)


def fixture_root(tmp_path):
    for name in ['utils','models','data','scripts','docs']:(tmp_path/name).mkdir()
    for name in ['train.py','eval.py','tune.py','README.md','AGENTS.md','utils/README.md']:(tmp_path/name).write_text('')
    return tmp_path


def test_clean_repository_contract(tmp_path):
    assert audit_module.audit(fixture_root(tmp_path))==[]


def test_generated_outputs_and_root_probe_are_rejected(tmp_path):
    r=fixture_root(tmp_path);(r/'probe.py').write_text('');(r/'scripts/generated.csv').write_text('')
    rules={x['rule'] for x in audit_module.audit(r)}
    assert {'root_allowlist','generated_product_in_scripts'}<=rules


def test_retired_and_reverse_imports_are_rejected(tmp_path):
    r=fixture_root(tmp_path)
    (r/'train.py').write_text('from budding_yeast_v2.utils.archive.online import io')
    (r/'utils/bad.py').write_text('import budding_yeast_v2.scripts.analysis.some_tool\nfrom ..archive.online import contracts')
    (r/'utils/_compat.py').write_text("ALIASES={'old':'budding_yeast_v2.utils.archive.online.io'}")
    rules={x['rule'] for x in audit_module.audit(r)}
    assert {'retired_runtime_dependency','reverse_script_dependency','retired_compat_target'}<=rules


def test_current_runtime_never_imports_retired_snapshots():
    import subprocess,sys,os
    code="""
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.startswith('budding_yeast_v2.utils.archive'):raise AssertionError(name)
sys.meta_path.insert(0,Block())
import budding_yeast_v2.train
from budding_yeast_v2.utils.sampling import ActiveLearningEngine
from budding_yeast_v2.utils.sampling.online import fission
assert ActiveLearningEngine.__module__.startswith('budding_yeast_v2.utils.sampling.online')
assert not any(n.startswith('budding_yeast_v2.utils.archive') for n in sys.modules)
"""
    subprocess.run([sys.executable,'-c',code],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))


def test_retired_namespace_is_rejected_without_executing_originals():
    import importlib,pytest
    import budding_yeast_v2.utils
    with pytest.raises(ImportError,match='Retired archive code'):
        importlib.import_module('budding_yeast_v2.utils.archive.online.engine')

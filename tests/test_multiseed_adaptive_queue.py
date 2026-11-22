"""One owner for all adaptive seeds; protocols/results must never cross seeds."""
from dataclasses import asdict,replace
import json
from pathlib import Path
import pytest

from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig
from budding_yeast_v2.scripts.launch.queue_protocols import expand_seeds,verify_protocols,validate_adoption_tasks
from budding_yeast_v2.scripts.launch.run_sampling_extensions import tasks,command,main


def protocols(tmp_path):
    for seed in (42,123,1024):
        d=tmp_path/f'seed_{seed}';d.mkdir()
        value=dict(config=asdict(replace(BenchmarkConfig(),seed=seed)),fingerprint=f'hash{seed}',
            protocol_id=f'seed{seed}',pools={'mutation':{'sha256':'fixed-mutation'},'lhs':{'sha256':'fixed-lhs'}},
            real_reference={'sha256':'fixed-real'})
        (d/'protocol.json').write_text(json.dumps(value))
    return tmp_path/'seed_42'


def test_joint_expansion_preserves_live_ids_and_routes_each_protocol(tmp_path):
    root=protocols(tmp_path);original=[dict(r,output='/results',seed=42) for r in tasks('adaptive')]
    rows,receipts=expand_seeds(original,root,[42,123,1024])
    assert len(rows)==len({r['task_id'] for r in rows})==9
    assert [r['task_id'] for r in rows[:3]]==[r['task_id'] for r in original]
    for row in rows:
        cmd=command(row,Path(row['protocol']),'/unused')
        assert cmd[cmd.index('--protocol')+1].endswith(f"seed_{row['seed']}")
    verify_protocols(receipts)
    validate_adoption_tasks(rows,original,extend=True)
    with pytest.raises(ValueError):validate_adoption_tasks(rows,original)
    bad=[dict(r) for r in rows];bad[0]['seed']=123
    with pytest.raises(ValueError):validate_adoption_tasks(bad,original,extend=True)
    p=tmp_path/'seed_123/protocol.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError):verify_protocols(receipts)


def test_mismatched_training_or_pool_cannot_be_combined(tmp_path):
    root=protocols(tmp_path);p=tmp_path/'seed_123/protocol.json';v=json.loads(p.read_text())
    v['config']['lr']=.9;p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='training'):expand_seeds(tasks('adaptive'),root,[42,123])
    v['config']['lr']=BenchmarkConfig().lr;v['pools']['mutation']['sha256']='other';p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='pool/reference'):expand_seeds(tasks('adaptive'),root,[42,123])
    with pytest.raises(ValueError):expand_seeds(tasks('adaptive'),root,[42,42])


def test_multiseed_dry_run_only_nine_new_methods_and_authorized_cards(tmp_path,capsys):
    root=protocols(tmp_path)
    main(['--suite','adaptive','--seeds','42','123','1024','--protocol',str(root),
        '--directory',str(tmp_path/'absent'),'--gpus','0','1','2','3','--initial-jobs','8',
        '--max-jobs','8','--jobs-per-gpu','2','--dry-run'])
    plan=json.loads(capsys.readouterr().out)
    assert plan['count']==9 and plan['gpus']==['0','1','2','3']
    assert {r['seed'] for r in plan['tasks']}=={42,123,1024}
    assert not (tmp_path/'absent').exists()


def test_staged_final_only_prediction_plot_policy_preserved():
    root=Path(__file__).resolve().parents[1]
    for file in ['utils/sampling/generation/runner.py','utils/sampling/generation/hggs_runner.py']:
        source=(root/file).read_text()
        assert 'plots=plots and (i == len(cfg.budgets) - 1)' in source
    source=(root/'utils/sampling/benchmark/runner.py').read_text()
    assert 'plots=plots and (round_index == len(config.budgets) - 1)' in source

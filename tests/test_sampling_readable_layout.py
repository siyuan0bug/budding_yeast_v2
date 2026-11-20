"""Readable paths, shared splits and lossless same-filesystem migration."""
from dataclasses import replace
import json
from pathlib import Path
from budding_yeast_v2.utils.sampling.layout import experiment_directory
from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
from budding_yeast_v2.scripts.maintenance.migrate_sampling_layout import move
from budding_yeast_v2.scripts.launch.run_sampling_extensions import tasks,find_result,command
from test_sampling_benchmark import config,make_pool


def test_layout_separates_model_pool_workflow_and_seed():
    cfg=config();p=experiment_directory('/out',cfg,'mutation','lcmd_ll')
    assert p.parts[:5]==('/','out','fno','mutation','lcmd')
    assert p.name=='seed_42' and 'w4_n1_m2_b4' in p.parent.name
    assert experiment_directory('/out',replace(cfg,seed=2026),'mutation','lcmd_ll').name=='seed_2026'
    a=experiment_directory('/out',cfg,'mutation','ours_intervention_closure')
    b=experiment_directory('/out',cfg,'mutation','ours_intervention_closure','generate')
    assert a!=b and a.parts[-3]=='ours_closure_pool' and b.parts[-3]=='ours_closure_generate'


def test_training_configuration_reuses_split_inodes(tmp_path):
    pool,_=make_pool(tmp_path/'pool');cfg=config()
    first=prepare_protocol({'mutation':pool,'lhs':pool},tmp_path/'shared',cfg)
    second=prepare_protocol({'mutation':pool,'lhs':pool},tmp_path/'shared',replace(cfg,width=8))
    assert first.name=='seed_42' and second!=first
    for name in ['mutation_split.npz','lhs_inputs.npz']:
        assert (first/name).stat().st_ino==(second/name).stat().st_ino
    assert json.loads((first/'protocol.json').read_text())['config']['width']==4
    assert json.loads((second/'protocol.json').read_text())['config']['width']==8


def test_move_preserves_hash_references_and_relative_links(tmp_path):
    old=tmp_path/'old';old.mkdir();(old/'file').write_bytes(b'original')
    (old/'link').symlink_to('file');inode=(old/'file').stat().st_ino
    new=tmp_path/'deeper/new';move(old,new)
    assert old.is_symlink() and (old/'file').read_bytes()==b'original'
    assert (new/'file').stat().st_ino==inode and (new/'link').read_bytes()==b'original'
    assert move(old,new)['reused']


def test_all_suite_and_discovery_use_one_root(tmp_path):
    rows=tasks('all');assert len(rows)==len({r['task_id'] for r in rows})==23
    task=next(r for r in rows if r['method']=='lcmd_ll' and r['pool']=='mutation')
    task.update(output=str(tmp_path),seed=2026)
    d=experiment_directory(tmp_path,replace(config(),seed=2026),'mutation','lcmd_ll')
    d.mkdir(parents=True);(d/'config.json').write_text('{}')
    assert find_result(task,'internal hash ignored')==str(d)
    cmd=command(task,'protocol','spec');assert cmd[cmd.index('--seed')+1]=='2026'

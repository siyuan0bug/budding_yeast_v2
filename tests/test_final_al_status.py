"""A successful retry must supersede an older failure at the same run path."""
import json
from budding_yeast_v2.scripts.analysis.dataset_diagnosis import compare_al_range_experiments as audit


def test_newer_success_overrides_older_failure_independent_of_directory_creation(tmp_path,monkeypatch):
    root=tmp_path/'al_budding/fno'
    run=root/'strategy/settings_refit_gene_ranges/seed_42';run.mkdir(parents=True)
    args=dict(num_al_rounds=4,al_strategy='ours_response',seed=42)
    (run/'config.json').write_text(json.dumps({'args':args}))
    (run/'train.log').write_text('completed')
    (run/'summary').mkdir();(run/'summary/rounds_summary.json').write_text(json.dumps([{}]*4))
    for stamp,code in [('20261029_153354',0),('20261029_023050',143)]:
        path=root.parent/'batch_logs'/stamp/'gpu0.tsv';path.parent.mkdir(parents=True)
        path.write_text(f'seed\tmodel\tstrategy\texit_code\texperiment_dir\n42\tfno\tours_response\t{code}\t{run}\n')
    monkeypatch.setattr(audit,'active_runs',lambda:{})
    rows,_=audit.discover(root)
    assert rows[0][0]['status']=='completed' and rows[0][0]['exit_code']==0

"""Shared strict checkpoint/data loading for downstream model comparisons."""
import torch
from torch.utils.data import DataLoader

from budding_yeast_v2.eval import load_model_from_checkpoint
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.sensitivity import preprocessing_fingerprint


def predict_real_checkpoints(checkpoints,dataset_path,device='cpu',batch_size=32):
    """Evaluate explicit LABEL=PATH inputs with identical preprocessing/cohorts.

    Incompatible normalization is rejected rather than mixing normalized inputs
    from one model with another model's statistics. No architecture fallback.
    """
    entries={}
    for item in checkpoints:
        label,separator,path=item.partition('=')
        if not separator or not label or not path or label in entries:
            raise ValueError('Each checkpoint must be a unique LABEL=PATH')
        entries[label]=path
    if not entries:
        raise ValueError('At least one checkpoint is required')
    predictions={}; target=None; reference_dm=None; fingerprint=None
    for label,path in entries.items():
        model,hp=load_model_from_checkpoint(path,device=device)
        dm=YeastDataModule(dataset_path=dataset_path,num_workers=0,use_adj_matrix=False,
            norm_type=hp['norm_type'],seed=hp['data_seed'],fit_scope=hp['normalization_fit_scope'],
            preprocessing_state=model._preprocessing_state)
        dm.setup('fit')
        current=preprocessing_fingerprint(dm)
        if fingerprint is not None and current != fingerprint:
            raise ValueError('Checkpoint normalization differs; evaluate models separately with eval.py')
        if reference_dm is not None and list(dm.test_indices[1]) != list(reference_dm.test_indices[1]):
            raise ValueError('Checkpoint evaluation cohorts differ')
        fingerprint=current;reference_dm=dm
        model.to(device).eval();model.bind_datamodule_stats(dm)
        outputs=[];targets=[]
        with torch.no_grad():
            for x,p,y in DataLoader(dm.test_dataset_real,batch_size=batch_size,shuffle=False):
                x,p,y=x.to(device),p.to(device),y.to(device)
                outputs.append(dm.denormalize_y(model(x,p),device).cpu())
                targets.append(dm.denormalize_y(y,device).cpu())
        if not outputs:
            raise ValueError('Dataset has no real-mutant evaluation samples')
        predictions[label]=torch.cat(outputs)
        current_target=torch.cat(targets)
        if target is not None:
            torch.testing.assert_close(target,current_target)
        target=current_target
    names=[str(reference_dm._mutant_names[i]) for i in reference_dm.test_indices[1]]
    return predictions,target,names

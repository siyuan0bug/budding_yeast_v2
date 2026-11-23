"""Fixed-update warm-start diagnostics through the existing Lightning model.

No loss, optimizer-update loop, normalization or evaluation is implemented here.
Callers supply a model factory and an already prepared DataModule. The final
update is evaluated; no diagnostic probe labels choose a checkpoint.
"""
from pathlib import Path
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from budding_yeast_v2.data.dataset_archive import atomic_json
from .runner import make_trainer


class FixedStepAudit(Callback):
    def __init__(self):
        self.history = []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group['params']):
            raise FloatingPointError('Nonfinite gradient in fixed-step diagnostic')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fixed-step training loss')
        self.history.append(dict(step=int(trainer.global_step), loss=float(loss.detach()),
                                 lr=float(trainer.optimizers[0].param_groups[0]['lr'])))
        if trainer.global_step % 50 == 0:
            print(f'[GAIN PROBE] step={trainer.global_step} loss={float(loss):.6g}', flush=True)


def fit_fixed_steps(model_factory, datamodule, output, *, steps, seed, device='cpu'):
    """Use an explicit finite DataLoader and exactly ``steps`` optimizer updates."""
    if steps < 1:
        raise ValueError('Positive optimizer update budget required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(seed, workers=True)
    audit = FixedStepAudit()
    model = model_factory()
    target = torch.device(device)
    gpu = target.type == 'cuda'
    started = time.monotonic()
    trainer = make_trainer(max_epochs=1, max_steps=steps, logger=False,
        accelerator='gpu' if gpu else 'cpu', devices=[target.index or 0] if gpu else 1,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        limit_val_batches=0, callbacks=[audit], deterministic=True)
    trainer.fit(model, datamodule=datamodule)
    if trainer.global_step != steps or len(audit.history) != steps:
        raise RuntimeError('Diagnostic failed to consume the exact optimizer update budget')
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite final diagnostic weights')
    trainer.save_checkpoint(output/'last.ckpt')
    atomic_json(output/'training.json', dict(steps=steps, seed=seed,
        seconds=time.monotonic()-started, history=audit.history,
        checkpoint_choice='fixed_final_update',
        gradient_clip_norm=1., optimizer_reset=True, shared_lightning_training_step=True))
    return model

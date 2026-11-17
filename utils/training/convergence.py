"""Loss-change diagnostics for fixed-budget AL training; never early-stop.

Only the training objective and synthetic validation MSE are inspected. Small
changes indicate an operational plateau, not a proof of optimizer stationarity.
"""
import csv
import json
from pathlib import Path

import numpy as np
from pytorch_lightning import Callback


def loss_rates(values):
    values=np.asarray(values,dtype=float)
    if len(values)<2 or not np.isfinite(values).all() or (values<0).any():
        return None
    if not np.any(values):
        return dict(net_pct=0.,trend_pct=0.,mean_absolute_step_pct=0.)
    if (values==0).any():return None # Relative change through zero is undefined.
    logs=np.log(np.maximum(values,np.finfo(float).tiny))
    steps=len(values)-1
    return dict(net_pct=float(-100*np.expm1((logs[-1]-logs[0])/steps)),
        trend_pct=float(-100*np.expm1(np.polyfit(np.arange(len(values)),logs,1)[0])),
        mean_absolute_step_pct=float(np.mean(np.abs(-100*np.expm1(np.diff(logs))))))


class LossChangeMonitor(Callback):
    """10-epoch trends, three consecutive checks; informational, no stopping.

    Check intervals do not overlap, except at their shared endpoint. Trends are
    fit to log(loss); absolute changes and endpoint rates remain separately visible.
    The 0.1%/epoch threshold is a declared experiment diagnostic, not universal.
    """
    def __init__(self,directory,window=10,rate_threshold_pct=.1,required_checks=3):
        super().__init__()
        if window<2 or required_checks<1 or not np.isfinite(rate_threshold_pct) or rate_threshold_pct<=0:
            raise ValueError('Invalid convergence diagnostic settings')
        self.directory=Path(directory)
        self.window=int(window);self.threshold=float(rate_threshold_pct);self.required=int(required_checks)
        self.history=[];self.records=[];self.streak=0;self.last_check_epoch=None

    def on_train_epoch_end(self,trainer,pl_module):
        if getattr(trainer,'sanity_checking',False):return
        epoch=int(trainer.current_epoch)+1
        metrics=trainer.callback_metrics
        value=metrics.get('train_loss',metrics.get('train_loss_epoch'))
        if value is None:return
        validation=metrics.get('val_loss_synth')
        train=float(value);val=float(validation) if validation is not None else float('nan')
        lr=float(trainer.optimizers[0].param_groups[0]['lr'])
        if self.history and epoch<=self.history[-1]['epoch']:
            raise ValueError('Convergence epoch order is not increasing')
        self.history.append(dict(epoch=epoch,train=train,val=val,lr=lr))
        if len(self.history)<self.window+1:return
        recent=self.history[-self.window-1:]
        rates={name:loss_rates([row[name] for row in recent]) for name in ('train','val')}
        record=dict(epoch=epoch,lr=lr,train_loss=train if np.isfinite(train) else None,
                    val_mse=val if np.isfinite(val) else None)
        for name,rate in rates.items():
            for metric in ('net_pct','trend_pct','mean_absolute_step_pct'):
                record[name+'_'+metric]=None if rate is None else rate[metric]
        if epoch%self.window==0:
            plateau=all(rate is not None and abs(rate['trend_pct'])<self.threshold for rate in rates.values())
            self.streak=self.streak+1 if plateau else 0
            self.last_check_epoch=epoch
            def show(name):
                return 'unavailable' if rates[name] is None else f"{rates[name]['trend_pct']:+.4f}%/epoch"
            print(f'[Loss change] epoch={epoch} | train_trend={show("train")} | '
                  f'val_trend={show("val")} | lr={lr:.3e} | '
                  f'plateau_checks={self.streak}/{self.required} | diagnostic_only=True',flush=True)
        record['plateau_checks']=self.streak
        record['plateau_candidate']=self.streak>=self.required
        self.records.append(record)
        self.directory.mkdir(parents=True,exist_ok=True)
        with (self.directory/'convergence.csv').open('a',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(record))
            if len(self.records)==1:writer.writeheader()
            writer.writerow(record)
        logger=getattr(trainer,'logger',None)
        if logger:
            values={f'convergence/{k}':v for k,v in record.items() if v is not None and k!='epoch' and np.isfinite(v)}
            values['epoch']=int(trainer.current_epoch)
            logger.log_metrics(values,step=trainer.global_step)

    def on_train_end(self,trainer,pl_module):
        self.directory.mkdir(parents=True,exist_ok=True)
        final=self.records[-1] if self.records else None
        if final is None:status='insufficient_history'
        elif any(final[name+'_trend_pct'] is None for name in ('train','val')):status='invalid_or_missing_loss'
        elif self.streak>=self.required:status='plateau_candidate'
        else:status='still_changing'
        report=dict(status=status,diagnostic_only=True,early_stopping=False,
            window=self.window,threshold_pct_per_epoch=self.threshold,required_checks=self.required,
            consecutive_checks=self.streak,last_check_epoch=self.last_check_epoch,
            epochs_observed=len(self.history),planned_epochs=int(trainer.max_epochs),
            peak_lr=float(pl_module.lr),min_lr=float(pl_module.hparams.get('min_lr',1e-6)),
            loss_definitions=dict(train='training objective',val='synthetic validation MSE'),
            excluded_metrics=['synthetic test','real mutant test'],
            final=final,note='Small loss changes can also follow a small learning rate; not a convergence proof')
        encoded=json.dumps(report,indent=2,allow_nan=False)
        temporary=self.directory/'convergence.json.tmp'
        temporary.write_text(encoded);temporary.replace(self.directory/'convergence.json')
        print(f'[Loss change] training ended: {status}; epochs={len(self.history)}/{trainer.max_epochs}; '
              f'no automatic early stopping',flush=True)

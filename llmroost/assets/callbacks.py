import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.size']  = 14

class PrintLossCallback(TrainerCallback):
    "A callback that prints the training and evaluation loss every n steps"
    def __init__(self, print_step=100):
        self.print_step = print_step

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.print_step == 0:
            print(f"Step: {state.global_step}")
            print(f"Training loss: {state.log_history[-1]['loss']}")
            if 'eval_loss' in state.log_history[-1]:
                print(f"Validation loss: {state.log_history[-1]['eval_loss']}")

class PrintLosses(pl.Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss']
        val_loss   = trainer.callback_metrics['val_loss']

        print(f'Epoch {trainer.current_epoch}/{trainer.max_epochs}')
        print(f'Train loss: {train_loss:.4f} \t Val loss: {val_loss:.4f}')

class CheckGradients(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        grad_norms = []
        par_norms  = []
        for param in pl_module.parameters():
            par_norms.append(param.norm().item())
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
                
        mean_grad_norm = np.mean(grad_norms)
        mean_par_norm  = np.mean(par_norms)
        pl_module.log('mean_grad_norm', mean_grad_norm, on_step=True, on_epoch=True)
        pl_module.log('mean_par_norm', mean_par_norm, on_step=True, on_epoch=True)
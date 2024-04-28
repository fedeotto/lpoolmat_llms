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

class PlotEmbeddings(pl.Callback):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def on_train_epoch_end(self, trainer, pl_module):
        cur_epoch = trainer.current_epoch

        if cur_epoch % 10 != 0:
            datamodule = trainer.datamodule

            sys_ids = []
            forms   = []
            for batch in self.loader:
                sys_ids = sys_ids + batch.sys_id
                forms   = forms + batch.form
            
            #assign color to unique sys_ids
            unique_sys_ids = np.unique(sys_ids)
            colors  = np.linspace(0, 1, len(sys_ids))
            color_dict = dict(zip(sys_ids, colors))

            forms_colors = []
            forms_labels = []
            for i, (form, sys_id) in enumerate(zip(forms, sys_ids)):
                forms_colors.append(color_dict[sys_id])
                forms_labels.append(f'{sys_id}')

            df_info = pd.DataFrame({'form': forms,
                                    'sys_id': sys_ids,
                                    'color': forms_colors})
            embs = pl_module.test(self.loader)

            X    = PCA(n_components=2).fit_transform(embs)
            plt.scatter(X[:, 0], X[:, 1], c=forms_colors, label=forms_labels)
            plt.savefig(f'embeddings_epoch_{cur_epoch}.png')

        # Plot embeddings
        # plt.scatter(embs[:, 0], embs[:, 1], c=embs[:, 2])
        # plt.show(
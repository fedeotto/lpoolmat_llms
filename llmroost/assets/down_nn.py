import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

class MLP(nn.Module):
    def __init__(self, 
                 input_dim      = 200, 
                 hidden_dims    = [128,256],
                 act            = nn.ReLU,
                 output_dim     = 64):
        
        super().__init__()
        dims = [input_dim]+hidden_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([act() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

        self.acts      = nn.ModuleList([act() for _ in range(len(dims) - 1)])
        self.fc_out    = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, act in zip(self.fcs, self.acts):
            fea = act(fc(fea))
        return self.fc_out(fea)

class MLPLight(pl.LightningModule):
    def __init__(self, 
                 input_dim      = 200, 
                 hidden_dims    = [256, 512,128, 64],
                 act            = nn.ReLU,
                 output_dim     = 1,
                 batchnorm      = False,
                 lr             = 0.001):
        
        super().__init__()
        self.save_hyperparameters()

        dims = [input_dim]+hidden_dims
        self.lr        = lr
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([act() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

        self.acts      = nn.ModuleList([act() for _ in range(len(dims) - 1)])
        self.fc_out    = nn.Linear(dims[-1], output_dim)
        self.criterion = nn.L1Loss()

    def forward(self, fea):
        for fc, act in zip(self.fcs, self.acts):
            fea = act(fc(fea))
        return self.fc_out(fea)

    def test(self, loader):
        embs = []
        for data in loader:
            fea, _ = data
            fea    = fea.to(device)
            for fc, act in zip(self.fcs, self.acts[:-1]):
                fea = act(fc(fea))   
            fea = self.fcs[-1](fea)
            embs.append(fea.detach().cpu().numpy())    

        return np.vstack(embs)
     
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
                        'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=50, verbose=True),
                         'monitor': 'val_loss', 
                      }  
           
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        X, y  = batch
        X     = X.to(device)
        y     = y.to(device)
        
        preds = self.forward(X).squeeze()
        loss  = self.criterion(y,preds)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        X, y  = batch
        X     = X.to(device)
        y     = y.to(device)
        
        preds = self.forward(X).squeeze()
        loss  = self.criterion(y, preds)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def predict(self, test_loader):
        self.eval()
        preds = []
        for X_test, _ in test_loader:
            X_test = X_test.to(device)
            preds.append(self.forward(X_test).squeeze())
        preds = torch.hstack(preds).detach().cpu().numpy()

        return preds

class PrintResNetLoss(pl.Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss']
        val_loss   = trainer.callback_metrics['val_loss']

        print(f'Epoch {trainer.current_epoch}/{trainer.max_epochs}')
        print(f'train_loss: {train_loss:.4f} , \t val_loss: {val_loss:.4f}')

def downstream_preds_nn(X_train, 
                        y_train,
                        X_val,  #using val that crabnet & roost see during training.
                        y_val,
                        X_test,
                        y_test,
                        cfg:DictConfig
                        # dataset_name,
                        # batch_size  = 128,
                        # random_state= 1234,
                        # n_epochs    = 1000
                        ):
    
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    
    fc_net  = MLPLight(input_dim      = X_train.shape[1],
                        hidden_dims   = [512,256,128,64],
                        output_dim    = 1)
    
    model_checkpoint = ModelCheckpoint(dirpath  = './mlp_run_checkpoints',
                                        filename= f'mlp_{cfg.data.name}_{cfg.random_seed}',
                                        save_top_k=1,
                                        mode='min', 
                                        monitor="val_loss")
    
    trainer  = pl.Trainer(accelerator=device.type,
                          max_epochs=cfg.train.pl_trainer.max_epochs,
                        #   deterministic=True,
                          callbacks=[PrintResNetLoss(),
                                     LearningRateMonitor(),
                                     model_checkpoint,
                                     EarlyStopping(monitor="val_loss", 
                                                   mode='min',
                                                   patience=80, 
                                                   verbose=True)
                                                   ])
    
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    X_val   = torch.as_tensor(X_val, dtype=torch.float32)
    y_val   = torch.as_tensor(y_val, dtype=torch.float32)
    X_test  = torch.as_tensor(X_test, dtype=torch.float32)
    y_test  = torch.as_tensor(y_test, dtype=torch.float32)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val   = X_val.to(device)
    y_val   = y_val.to(device)
    X_test  = X_test.to(device)
    y_test  = y_test.to(device)

    train_dataset= TensorDataset(X_train, y_train)
    val_dataset  = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    torch.manual_seed(cfg.random_seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg.model.batch_size)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    trainer.fit(fc_net, train_dataloaders=train_loader,val_dataloaders=val_loader)
    fc_net = fc_net.load_from_checkpoint(model_checkpoint.best_model_path)
    fc_net = fc_net.to(device) #need to mover to device before predicting.
    preds  = fc_net.predict(test_loader)   

    return preds

class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network (copied from Roost Repository)
    """

    def __init__(self, 
                 internal_elem_dim, 
                 output_dim, 
                 hidden_layer_dims, 
                 batchnorm=False, 
                 negative_slope=0.2):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [internal_elem_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([nn.LeakyReLU(negative_slope) for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        return  self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__
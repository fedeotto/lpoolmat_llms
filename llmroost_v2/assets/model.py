import torch
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
from assets.down_nn import ResidualNetwork
import pytorch_lightning as pl
import pandas as pd
from torch_geometric.data import Batch
import math
from transformers import LlamaTokenizer, LlamaForCausalLM
import warnings
from sklearn.metrics import mean_absolute_error, r2_score
from assets.roost_encoder import RoostEncoder
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import hydra

warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, 
            params=self.parameters())
        
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt)
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}

class LLMRoost(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # torch.manual_seed(self.hparams.random_state)

        self.embedder      = nn.Linear(self.hparams.model.input_size, 
                                       self.hparams.model.in_channels-1)
        self.llm           = hydra.utils.instantiate(self.hparams.model.llm.type, _recursive_=False)
        
        self.tokenizer     = hydra.utils.instantiate(self.hparams.model.llm.tokenizer, _recursive_=False)
        self.encoder       = hydra.utils.instantiate(self.hparams.model.encoder, _recursive_=False)

        if self.hparams.model.agg_type == 'concat':
            #double to input the concatenated embeddings
            self.hparams.model.resnet.internal_elem_dim = self.hparams.model.resnet.internal_elem_dim*2
        else:
            self.hparams.model.resnet.internal_elem_dim = self.hparams.model.resnet.internal_elem_dim

        self.resnet        = hydra.utils.instantiate(self.hparams.model.resnet, _recursive_=False)
        self.criterion     = nn.L1Loss()

        self.projection    = nn.Sequential(nn.Linear(self.hparams.model.llm.emb_dim, self.hparams.model.llm.emb_dim//2),
                                        nn.ReLU(),
                                        nn.Linear(self.hparams.model.llm.emb_dim//2, self.hparams.model.llm.emb_dim//4),
                                        nn.ReLU(),
                                        nn.Linear(self.hparams.model.llm.emb_dim//4, self.hparams.model.internal_elem_dim))           
                                        
        #ensuring LLM weights are frozen
        for param in self.llm.parameters():
            param.requires_grad = False

    def forward(self, batch, nbrs_prompts=None, stage='train'):
        #LLM part
        if stage == 'train':
            # Extract the (train) context information
            text_data = batch.prompt
            # Tokenize the text data and pass it to the language model
            inputs = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
            # Move the inputs to the GPU
            inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}
            outputs       = self.llm(**inputs)
            lm_embeddings = outputs.hidden_states[-1]
            lm_embeddings = lm_embeddings.mean(dim=1)
        
        elif stage == 'val' or stage == 'test':
            avg_embs = []
            for nbr_prompt in nbrs_prompts:
                # Tokenize the text data and pass it to the language model
                inputs = self.tokenizer(nbr_prompt, return_tensors='pt', padding=True, truncation=True)

                # Move the inputs to the GPU
                inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}
                outputs= self.llm(**inputs)
                lm_embeddings = outputs.hidden_states[-1]
                lm_embeddings = lm_embeddings.mean(dim=1).mean(dim=0)
                # lm_embeddings = lm_embeddings.mean(dim=0)
                avg_embs.append(lm_embeddings)
            lm_embeddings = torch.vstack(avg_embs)

        lm_proj   = self.projection(lm_embeddings)

        #Roost part
        embeds    = self.embedder(batch.x)
        embeds    = torch.cat([embeds, batch.weights], dim=1)
        encodings = self.encoder(embeds, batch.edge_index, batch.weights.squeeze(-1), batch_index=batch.batch)

        #Normalizing encodings
        encodings = F.normalize(encodings, dim=1, p=2)

        #normalizing llm embeddings
        lm_proj   = F.normalize(lm_proj, dim=1, p=2)

        if self.hparams.model.agg_type == 'concat':
            encodings = torch.cat([encodings, lm_proj], dim=1)
        elif self.hparams.model.agg_type == 'sum':
            encodings = encodings + lm_proj

        preds = self.resnet(encodings)
        return preds

    def training_step(self, batch, batch_idx):
        stage = 'train'
        batch = batch.to(device)
        y     = batch.y
        preds = self(batch, nbrs_prompts=None, stage=stage)
        loss  = self.compute_loss(y, preds, stage=stage)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        stage = 'val'
        train_formulas = self.trainer.datamodule.df_train['formula'].tolist()
        val_formulas   = batch.formula #val_formulas in the batch
        df_train       = self.trainer.datamodule.df_train #needed to extract train context
        lookup         = self.trainer.datamodule.lookup

        #Computing nearest neighbors for formulas in the validation set
        val_train_lookup = lookup.loc[val_formulas,train_formulas]
        sorted_indices   = np.argsort(val_train_lookup, axis=1)[:,1:self.trainer.datamodule.K+1]
        nbrs_2d          = np.vectorize(val_train_lookup.columns.__getitem__)(sorted_indices)

        all_nbrs_prompts = []
        for nbr in nbrs_2d:
            nbrs_prompts = []
            struc_fam = df_train.loc[df_train['formula'].isin(nbr),'StructuralFamily'].values
            chem_fam  = df_train.loc[df_train['formula'].isin(nbr),'ChemicalFamily'].values

            for i, (fam1,fam2) in enumerate(zip(struc_fam,chem_fam)):
                nbrs_prompts.append(f'The structural family of {nbr[i]} is {fam1}. The chemical family of {nbr[i]} is {fam2}')
            all_nbrs_prompts.append(nbrs_prompts)

        batch  = batch.to(device)
        y      = batch.y
        preds  = self(batch, nbrs_prompts=all_nbrs_prompts, stage=stage)

        loss = self.compute_loss(y, preds, stage=stage)
        return loss

    def test_step(self, batch, batch_idx, stage='test'):
        stage='test'
        train_formulas = self.trainer.datamodule.df_train['formula'].tolist()
        test_formulas  = batch.formula #test_formulas in the batch
        df_train       = self.trainer.datamodule.df_train #needed to extract train context
        lookup         = self.trainer.datamodule.lookup

        #Computing nearest neighbors for formulas in the test set
        test_train_lookup= lookup.loc[test_formulas,train_formulas]
        sorted_indices   = np.argsort(test_train_lookup, axis=1)[:,1:self.trainer.datamodule.K+1]
        nbrs_2d          = np.vectorize(test_train_lookup.columns.__getitem__)(sorted_indices)

        all_nbrs_prompts = []
        for nbr in nbrs_2d:
            nbrs_prompts = []
            struc_fam = df_train.loc[df_train['formula'].isin(nbr),'StructuralFamily'].values
            chem_fam  = df_train.loc[df_train['formula'].isin(nbr),'ChemicalFamily'].values

            for i, (fam1,fam2) in enumerate(zip(struc_fam,chem_fam)):
                nbrs_prompts.append(f'The structural family of {nbr[i]} is {fam1}. The chemical family of {nbr[i]} is {fam2}')
            all_nbrs_prompts.append(nbrs_prompts)

        batch  = batch.to(device)
        y      = batch.y
        preds  = self(batch, nbrs_prompts=all_nbrs_prompts, stage=stage)

        y     = y.cpu().numpy()
        preds = preds.detach().cpu().numpy()

        mae    = mean_absolute_error(y, preds)
        r2     = r2_score(y, preds)

        self.log_dict({f'{stage}_mae': mae,
                        f'{stage}_r2': r2}, on_step=True, on_epoch=True, prog_bar=True)

    def predict(self, loader):
        self.eval()
        preds = []
        y     = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = self(batch)
                preds.append(output.cpu().numpy())
                y.append(batch.y.cpu().numpy())

        return (np.concatenate(y,axis=0), np.concatenate(preds, axis=0))
        
    def compute_loss(self, y, preds, stage='train'):
        loss       = self.criterion(y,preds)
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


class Roost(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embedder      = nn.Linear(self.hparams.model.input_size, 
                                       self.hparams.model.in_channels-1)
        self.encoder       = hydra.utils.instantiate(self.hparams.model.encoder)
        self.resnet        = hydra.utils.instantiate(self.hparams.model.resnet)
        self.criterion     = nn.L1Loss()
        
    def forward(self, batch):
        embeds    = self.embedder(batch.x)
        embeds    = torch.cat([embeds, batch.weights], dim=1)
        encodings = self.encoder(embeds, batch.edge_index, batch.weights.squeeze(-1), batch_index=batch.batch)
        preds     = self.resnet(encodings)

        return preds

    def training_step(self, batch, batch_idx):
        batch  = batch.to(device)
        y      = batch.y
        preds  = self(batch)

        loss = self.compute_loss(y, preds, stage='train')
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch  = batch.to(device)
        y      = batch.y
        preds  = self(batch)

        loss = self.compute_loss(y, preds, stage='val')
        return loss

    def test_step(self, batch, batch_idx, stage='test'):
        batch  = batch.to(device)
        y      = batch.y
        preds  = self(batch)

        y     = y.cpu().numpy()
        preds = preds.detach().cpu().numpy()

        mae        = mean_absolute_error(y, preds)
        r2         = r2_score(y, preds)

        self.log_dict({f'{stage}_mae': mae,
                        f'{stage}_r2': r2}, on_step=True, on_epoch=True, prog_bar=True)

    def predict(self, loader):
        self.eval()
        preds = []
        y     = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = self(batch)
                preds.append(output.cpu().numpy())
                y.append(batch.y.cpu().numpy())
        return (np.concatenate(y,axis=0), np.concatenate(preds, axis=0))
        
    def compute_loss(self, y, preds, stage='train'):
        loss       = self.criterion(y,preds)
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
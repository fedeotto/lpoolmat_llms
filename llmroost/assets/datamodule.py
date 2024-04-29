"""
Created on Wed Apr 17 17:10:13 2024

@author: federico
"""

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from common.utils import MMScaler
import torch
from assets.data import create_material_data
import pandas as pd
from omegaconf import DictConfig

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 df_train=None,
                 df_val=None,
                 df_test=None,
                 desc : str='' ,
                 elem_prop : str = 'mat2vec',
                 batch_size=32,
                 random_state=0):
        
        super().__init__()
        torch.manual_seed(random_state)

        self.elem_prop     = pd.read_csv(f'./assets/element_properties/{elem_prop}.csv',index_col=0)
        self.df_train      = df_train
        self.df_val        = df_val
        self.df_test       = df_test
        self.batch_size    = batch_size
        self.desc          = desc

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = create_dataset(self.df_train, 
                                                elem_attrs=self.elem_prop,
                                                desc=self.desc)
            
            self.val_dataset   = create_dataset(self.df_val, 
                                                elem_attrs=self.elem_prop,
                                                desc=self.desc)
        
        if stage == 'test' or stage is None:
            self.test_dataset = create_dataset(self.df_test,
                                               elem_attrs=self.elem_prop,
                                               desc=self.desc)

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size.train, shuffle=True)
    
    def return_traindataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size.train, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size.val, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size.test, shuffle=False)

def create_dataset(df, elem_attrs,desc):
    """
    Create a dataset from a dataframe.
    """
    comp_graphs = [] #list of pyg Data objects
    for i, (f,t) in enumerate(zip(df['formula'],df['target'])):
        comp_graphs.append(create_material_data(f,t,elem_attrs=elem_attrs,desc=desc))
    return comp_graphs
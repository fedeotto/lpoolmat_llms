"""
Created on Tue Feb 21 14:15:13 2023

@author: federico
"""
import pandas as pd
import common.chem as chem
import numpy as np
import os
# from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

def load_dataset(dataset_name  : str  = 'agl_thermal_conductivity_300K',
                 elem_prop     : str  = 'mat2vec',
                 bench_type    : str  = 'ood',
                 random_state  : int  = 1234):
        
    print(f'\n --- Loading {dataset_name} ---\n')

    df = pd.read_excel(f'datasets/{dataset_name}.xlsx')
    df = preprocess_dataset(df,
                            dataset_name = dataset_name,
                            elem_prop    = elem_prop)
    
    df = df.dropna()
    df_train, df_val, df_test = create_bench(df            = df, 
                                            random_state  = random_state,
                                            bench_type    = bench_type)
    
    return (df_train, df_val, df_test)

def create_bench(df            : pd.DataFrame,
                 random_state  : int   = 1234,
                 bench_type    : str   = 'id'):
    
    train_size = int(len(df)*0.7)
    val_size   = int(len(df)*0.1)
    test_size  = int(len(df)*0.2)

    df_train = df.sample(n=train_size, random_state=random_state)
    df = df.drop(index=df_train.index)
    df_val   = df.sample(n=val_size, random_state=random_state)
    df = df.drop(index=df_val.index)
    df_test  = df.sample(n=test_size, random_state=random_state)

    assert set(df_train['formula']) & set(df_val['formula']) & set(df_test['formula']) == set(), 'Found same points in both train and val/test sets!'

    # df_train = df_train[['formula','target']]
    # df_val   = df_val[['formula','target']]
    # df_test  = df_test[['formula','target']]

    return (df_train, df_val, df_test)

def preprocess_dataset(df,
                       dataset_name : str = 'agl_thermal_conductivity_300K',
                       elem_prop    : str = 'mat2vec'):
    """
    Preprocessing of original df.
    """    
    elem_props  = pd.read_csv(f'assets/element_properties/{elem_prop}.csv', 
                             index_col='element')
    valid_elems = list(elem_props.index)

    try:
        df = df[['formula','target','struct','loco']]
    except:
        pass
    df = df.reset_index(drop=True)
    
    idxs_to_drop = []
    for i,f in enumerate(df['formula']):
        try:
            elems, _ = chem._fractional_composition_L(f)
            if len(set(elems) & set(valid_elems)) != len(elems):
                idxs_to_drop.append(i)
        except:
            idxs_to_drop.append(i)
    
    df = df.drop(index=idxs_to_drop).reset_index(drop=True)
    df = df.sample(frac=1, random_state=1234)
        
    return df


#Just using good ol' mat2vec
def create_material_data(form, target, elem_attrs,struct_fam, chem_fam):    
    elems, fracs = chem._fractional_composition_L(form)
    nele         = len(elems)                
    self_fea_idx = []
    nbr_fea_idx  = []

    # prompt = f'What is the {desc} of {form}?'
    prompt = f'The structural family of {form} is {struct_fam}. The chemical family of {form} is {chem_fam}'
        
    for j, _ in enumerate(elems):
        self_fea_idx += [j] * nele
        nbr_fea_idx += list(range(nele))
            
    self_fea_idx = torch.tensor([self_fea_idx], dtype=torch.long)
    nbr_fea_idx = torch.tensor([nbr_fea_idx], dtype=torch.long)
        
    edge_index = torch.cat([nbr_fea_idx, self_fea_idx], dim=0)        
    weights = torch.tensor(fracs, dtype=torch.float32).view(-1,1)
    target = torch.tensor([[target]], dtype=torch.float32)
    
    x      = torch.tensor(elem_attrs.loc[elems].values, dtype=torch.float32)

    comp_graph = Data(x=x, 
                      formula   = form,
                      prompt    = prompt,
                      edge_index= edge_index,
                      weights   = weights,
                      y         = target)
        
    return comp_graph
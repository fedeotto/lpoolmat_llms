import pandas as pd
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch
from common.chem import _element_composition

class Scaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = (data - self.mean) / self.std
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled)
        data = data_scaled * self.std + self.mean
        return data

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
class SDScaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = (data - self.mean) / self.std
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled)
        data = data_scaled * self.std + self.mean
        return data

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class MMScaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.max = torch.max(self.data)
        self.min = torch.min(self.data)
        
    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = (data - self.min) / (self.max - self.min)
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled)
        data = data_scaled * (self.max - self.min) + self.min
        return data

    def state_dict(self):
        return {'min': self.min,
                'max': self.max}

    def load_state_dict(self, state_dict):
        self.min = state_dict['min']
        self.max = state_dict['max']



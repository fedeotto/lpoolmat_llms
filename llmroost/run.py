'''author: fedeotto'''
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
from assets.data import load_dataset
from assets.datamodule import DataModule
from hydra.core.hydra_config import HydraConfig
from assets.model import LLMRoost, Roost
import pytorch_lightning as pl
from common.utils import load_envs
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

os.environ["HYDRA_FULL_ERROR"] = '1'

@hydra.main(config_path="conf", config_name="default")
def run(cfg: DictConfig):
    os.chdir(os.environ['PROJECT_ROOT'])
    
    hydra_dir    = Path(HydraConfig.get().sweep.dir) #change to .run.dir for single run
    hydra_subdir = Path(HydraConfig.get().sweep.subdir)
    
    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir/ hydra_subdir / "hparams.yaml").write_text(yaml_conf)

    pl.seed_everything(cfg.random_state)
    np.random.seed(cfg.random_state)

    # Load the dataset
    (df_train,
    df_val,df_test) = load_dataset(dataset_name = cfg.data.name,
                                    elem_prop   = 'mat2vec',
                                    random_state= cfg.random_state,
                                    bench_type  = cfg.bench_type)
    
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, 
                                         df_train=df_train,
                                         df_val  =df_val,
                                         df_test =df_test,
                                         _recursive_=False)
    if cfg.model.name == 'llmroost':
        model = LLMRoost(model   = cfg.model,
                        data     = cfg.data,
                        optim    = cfg.optim,
                        logging  = cfg.logging)
    else:
        model = Roost(model      = cfg.model,
                        data     = cfg.data,
                        optim    = cfg.optim,
                        logging  = cfg.logging)

    name = f'{cfg.expname}-{cfg.random_state}'
    group = f'{cfg.expname}'
    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(name=name,
                                    group=group,
                                    **wandb_config)
        wandb_logger.watch(model,
                            log=cfg.logging.wandb_watch.log,
                            log_freq=cfg.logging.wandb_watch.log_freq)
        
    trainer = pl.Trainer(max_epochs=cfg.train.pl_trainer.max_epochs,
                        gpus=cfg.train.pl_trainer.gpus,
                        deterministic=True,
                        logger=wandb_logger,
                        gradient_clip_val=cfg.train.pl_trainer.gradient_clip_val,
                        gradient_clip_algorithm=cfg.train.pl_trainer.gradient_clip_algorithm,
                        callbacks=[ModelCheckpoint(monitor=cfg.train.monitor_metric,
                                                    dirpath=hydra_dir/ hydra_subdir,
                                                    save_top_k=1,
                                                    mode='min',
                                                    filename=name),
                                    EarlyStopping(monitor=cfg.train.monitor_metric,
                                                  mode='min',
                                                    patience=cfg.train.early_stopping.patience)])
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')

    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    run()
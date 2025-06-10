import os
import hydra
import wandb
import numpy as np
import omegaconf
import lightning as L
import plotly.graph_objects as go

from datetime import datetime
from socket import gethostname

import torch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from src.TSAEDataset import TSAEDataset
from src.LearnedPoolSAE import LearnedPoolSAE
from src.LearnedPoolSVAE import LearnedPoolSVAE
from src.SpectralAEModule import SpectralAEModule
from src.utils import get_data_split, custom_collate_fn, custom_collate_fn_augmented_batch

# Typenames
DataLoaderTrain = DataLoader
DataLoaderVal = DataLoader
DataLoaderTest = DataLoader

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True


def build_lightning_module(cfg: omegaconf.DictConfig,
                           U_k_trn: torch.Tensor,
                           template_conn_trn: torch.Tensor,
                           U_k_val: torch.Tensor,
                           template_conn_val: torch.Tensor,
                           U_k_tst: torch.Tensor,
                           template_conn_tst: torch.Tensor,
                           task_name: str,
                           ) -> L.LightningModule:
    if cfg.training.model_type == 'AE':
        model = LearnedPoolSAE(k=cfg.training.k, size_latent=cfg.training.size_latent,
                               activation_func=cfg.training.activation_func,
                               initial_feature_size=cfg.training.initial_feature_size,
                               depth=cfg.training.depth,
                               device=cfg.training.device)
    elif cfg.training.model_type == 'VAE':
        model = LearnedPoolSVAE(k=cfg.training.k, size_latent=cfg.training.size_latent,
                                activation_func=cfg.training.activation_func,
                                initial_feature_size=cfg.training.initial_feature_size,
                                depth=cfg.training.depth,
                                device=cfg.training.device)
    else:
        raise NotImplementedError(f'Only AE and VAE models are available, but {cfg.training.model_type} provided.')

    module = SpectralAEModule(model=model, lr=cfg.training.lr,
                              U_k_trn=U_k_trn, template_conn_trn=template_conn_trn,
                              U_k_val=U_k_val, template_conn_val=template_conn_val,
                              U_k_tst=U_k_tst, template_conn_tst=template_conn_tst,
                              task_name=task_name,
                              kl_weight=cfg.training.kl_weight,
                              max_steps=cfg.training.max_steps)

    return module


@hydra.main(config_path='../configuration', config_name='config', version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    now = datetime.now()
    task_name = (f'{now.strftime("%b")}{now.strftime("%d")}_{now.strftime("%H")}'
                 f'-{now.strftime("%M")}-{now.strftime("%S")}_{gethostname()}')
    print(f'Preparing {task_name}...')

    ids_trn, ids_val, ids_tst = get_data_split(cfg)

    dataset_trn = TSAEDataset(cfg.paths.dataset, cfg.training.device, cfg.training.k, ids_trn,
                              apply_transform=cfg.training.denoising_training)
    dataset_val = TSAEDataset(cfg.paths.dataset, cfg.training.device, cfg.training.k, ids_val, apply_transform=False)
    dataset_tst = TSAEDataset(cfg.paths.dataset, cfg.training.device, cfg.training.k, ids_tst, apply_transform=False)

    loader_trn: DataLoaderTrain = DataLoader(dataset_trn, shuffle=True, batch_size=cfg.training.batch_size,
                                             # num_workers=cfg.training.num_workers,
                                             # persistent_workers=cfg.training.persistent_workers,
                                             collate_fn=lambda batch: custom_collate_fn(batch, apply_noise=cfg.training.denoising_training))
    loader_val: DataLoaderVal = DataLoader(dataset_val, shuffle=False, batch_size=cfg.training.batch_size,
                                           # num_workers=cfg.training.num_workers,
                                           # persistent_workers=cfg.training.persistent_workers,
                                           collate_fn=lambda batch: custom_collate_fn(batch, apply_noise=False))
    loader_test: DataLoaderTest = DataLoader(dataset_tst, shuffle=False, batch_size=1,
                                             # num_workers=cfg.training.num_workers,
                                             # persistent_workers=cfg.training.persistent_workers,
                                             collate_fn=lambda batch: custom_collate_fn(batch, apply_noise=False))

    model = build_lightning_module(cfg, dataset_trn.U_k, dataset_trn.templ_faces,
                                   dataset_val.U_k, dataset_val.templ_faces,
                                   dataset_tst.U_k, dataset_tst.templ_faces, task_name)

    cfg_full = dict(cfg.training)
    cfg_full['train_samples_ids'] = ids_trn
    cfg_full['validation_samples_ids'] = ids_val
    cfg_full['test_samples_ids'] = ids_tst

    wandb_logger = WandbLogger(project='ToothForge-GitHub-version', name=task_name, config=cfg_full)

    if not os.path.exists(os.path.join('logs/checkpoints', task_name)):
        os.makedirs(os.path.join('logs/checkpoints', task_name))
    if not os.path.exists(os.path.join('logs/media', task_name)):
        os.makedirs(os.path.join('logs/media', task_name))

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('logs/checkpoints', task_name), monitor="val_loss", mode="min",
        filename="best-checkpoint"
    )

    trainer = L.Trainer(accelerator=cfg.training.device,
                        max_steps=cfg.training.max_steps,
                        callbacks=[checkpoint_callback],
                        logger=wandb_logger,
                        precision=cfg.training.precision,
                        default_root_dir=os.path.join('logs/checkpoints', task_name),
                        num_sanity_val_steps=0,
                        gradient_clip_val=0.5)

    trainer.fit(model=model,
                train_dataloaders=loader_trn,
                val_dataloaders=loader_val,
                )
                # ckpt_path=cfg.paths.pretrained_model_path)  # Set this value to None to train from scratch.


if __name__ == '__main__':
    main()

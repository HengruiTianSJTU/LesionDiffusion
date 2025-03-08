import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy
from einops import rearrange
from glob import glob
import numpy as np
from natsort import natsorted

from ldm.modules.encoders.modules import TransformerEmbedder
from ldm.modules.diffusionmodules.model import Encoder
from ldm.modules.diffusionmodules.openaimodel import EncoderUNetModel, UNetModel
from ldm.util import log_txt_as_img, default, ismap, instantiate_from_config


class BasePytorchLightningTrainer(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        
    @torch.no_grad()
    def get_input(self, batch, k):
        return batch.get(k)
    
    @property
    def dataset_connector(self):
        return self.trainer._data_connector.trainer.datamodule.datasets

    def shared_step(self, batch):
        raise NotImplementedError()
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        if loss_dict is not None: self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("global_step", self.global_step, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        if loss_dict is not None: self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True, logger=True)
    
    @torch.no_grad()
    def log_images(self, batch, N=8, *args, **kwargs):
        raise NotImplementedError()
    
    def configure_optimizers(self):
        param = list(self.parameters())
        optimizer = AdamW(param, lr=self.learning_rate)
        return optimizer
        
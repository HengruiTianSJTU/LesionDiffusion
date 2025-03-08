import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from ldm.modules.vision_mamba import Vim
from ldm.modules.encoders.modules import TransformerEmbedder
from ldm.modules.diffusionmodules.openaimodel import EncoderUNetModel
from ldm.util import log_txt_as_img, default, ismap, instantiate_from_config, get_obj_from_str


class MaskTextClipModule(pl.LightningModule):

    def __init__(self,
                 image_encoder_config,
                 text_encoder_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 only_model=False,
                 image_key="image",
                 text_key="context",
                 use_scheduler=False,
                 scheduler_config=None,
                 monitor='val/loss',
                 *args,
                 **kwargs):
        super().__init__()
        self.image_key = image_key
        self.text_key = text_key
        self.monitor = monitor
        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        self.model = CLIPWrapper(image_encoder_config, text_encoder_config, **kwargs)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)
        
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
            
    def get_input(self, batch):
        image = batch.get(self.image_key)
        text = batch.get(self.text_key)
        return image, text
        
    def get_loss(self, v2t, t2v):
        prefix = "train" if self.training else "val"
        targets = torch.arange(v2t.shape[0], device=self.device)
        loss_v2t = F.cross_entropy(v2t, targets)
        loss_t2v = F.cross_entropy(t2v, targets)
        loss = loss_v2t + loss_t2v
        loss_dict = {f"{prefix}/loss_v2t": loss_v2t, f"{prefix}/loss_t2v": loss_t2v, f"{prefix}/loss": loss}
        return loss, loss_dict
        
    def shared_step(self, batch, t=None):
        image, text = self.get_input(batch)
        v2t, t2v = self.model(image, text)
        loss, loss_dict = self.get_loss(v2t, t2v)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self,):
        if self.use_scheduler:
            sch = self.lr_schedulers()
            sch.step()
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_parameters = list(self.model.parameters())
        opt = torch.optim.AdamW(opt_parameters, lr=lr)
        if self.use_scheduler:
            scheduler = get_obj_from_str(self.scheduler_config["target"])(opt, **self.scheduler_config["params"])
            cfg = {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "train/loss"
                }
            }
            return cfg
        return opt

    @torch.no_grad()
    def log_images(self, batch, N=8, *args, **kwargs):
        log = dict()
        v, t = self.get_input(batch)
        log["images"] = v[:N]
        texts = batch.get("text")[:N]
        log["texts"] = str(texts)
        
        pred_sims = self.model(log["images"], t[:N])[0].softmax(1)
        log["pred_sim_texts"] = str(np.array(texts)[pred_sims.argmax(1).cpu().numpy()].tolist()) + "\n" + str(pred_sims.amax(1))
        return log
        
        
class CLIPWrapper(nn.Module):
    def __init__(self, image_model_cfg, text_model_cfg, text_proj_in=5120*4):
        super().__init__()
        self.image_model = instantiate_from_config(image_model_cfg)
        self.text_model = instantiate_from_config(text_model_cfg)
        
        self.text_proj = nn.Linear(text_proj_in, self.image_model.num_classes)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, v, t):
        image_features = self.image_model(v)
        text_features = self.text_proj(self.text_model(t).view(v.shape[0], -1))
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
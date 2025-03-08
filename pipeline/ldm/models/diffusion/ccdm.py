import sys
import torch
import numpy as np
import torch.distributed
import torch.nn as nn
import torch.nn.functional as f

from tqdm import tqdm
from typing import Optional

from omegaconf import OmegaConf
from functools import partial
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from ldm.modules.diffusionmodules.util import make_beta_schedule

import pytorch_lightning as pl
from ldm.modules.ema import LitEma
from einops import repeat, rearrange
from contextlib import contextmanager
from ldm.modules.losses.lpips import LPIPS
from ldm.modules.diffusionmodules.util import extract_into_tensor
from ldm.modules.encoders.modules import OneHotCategoricalBCHW

sys.path.append("/mnt/workspace/dailinrui/code/multimodal/trajectory_generation/ccdm")
from ldm.util import get_obj_from_str, instantiate_from_config
from ldm.models.diffusion.ddim import make_ddim_timesteps


def identity(x, *args, **kwargs):
    return x


def default(x, dval=None):
    if not exists(x): return dval
    else: return x

def exists(x):
    return x is not None


class CategoricalDiffusion(pl.LightningModule):
    def __init__(self, unet_config, loss_config, *,
                 cond_stage_config=None,
                 train_ddim_sigmas=False,
                 data_key="mask",
                 cond_key="context",
                 timesteps=1000,
                 use_scheduler=False,
                 scheduler_config=None,
                 monitor=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 use_legacy=False,
                 noise_type="categorical",
                 load_only_unet=False,
                 conditioning_key="crossattn",
                 num_classes=12,
                 given_betas=None,
                 loss_type='l2',
                 beta_schedule="cosine",
                 linear_start=1e-2,
                 linear_end=2e-1,
                 cosine_s=8e-3,
                 use_ema=False,
                 class_weights=None,
                 p_x1_sample="majority",
                 parameterization="x0",
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 use_automatic_optimization=True,
                 dims=3,
                 **kwargs) -> None:
        super().__init__()
        self.dims = dims
        self.data_key = data_key
        self.cond_key = cond_key
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.conditioning_key = conditioning_key
        self.use_scheduler = use_scheduler
        self.loss_config = loss_config
        self.noise_type = noise_type
        self.train_ddim_sigmas = train_ddim_sigmas
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        if monitor is not None:
            self.monitor = monitor
        self.num_classes = num_classes
        self.automatic_optimization = use_automatic_optimization
        if not exists(class_weights):
            class_weights = torch.ones((self.num_classes,))
        self.register_buffer("class_weights", torch.tensor(class_weights))
        print(f"setting class weights as {self.class_weights}")
        self.use_ema = use_ema
        self.parameterization = parameterization
        self.cond_stage_forward = cond_stage_forward
        self.cond_stage_trainable = cond_stage_trainable
        
        self.loss_fn = dict()
        self.lpips = LPIPS().eval()
        self.p_x1_sample = p_x1_sample
        
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        
        unet_config["params"]["in_channels"] = self.num_classes
        unet_config["params"]["out_channels"] = self.num_classes
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet, use_legacy=use_legacy)
        self.dims = getattr(self.model, "dims", 3)
        
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
            
        if cond_stage_config is None: self.cond_stage_model = nn.Identity()
        else: self.cond_stage_model = instantiate_from_config(cond_stage_config)
        
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
                    
    def convert_legacy(self, state_dict):
        state_dict = state_dict["model"]
        
        self.timesteps = len(state_dict["betas"])
        self.betas = state_dict["betas"]
        self.alphas = 1 - state_dict["betas"]
        self.alphas_cumprod = state_dict["alphas_cumprod"]
        self.alphas_cumprod_prev = state_dict["alphas_cumprod_prev"]
        
        convert_dict = dict()
        legacy_keys = list(state_dict.keys())
        self_keys = list(self.model.state_dict().keys())
        
        for k in legacy_keys:
            if k.startswith("unet"):
                convert_dict[k.replace("unet", "model.diffusion_model")] = state_dict[k]
        return convert_dict
    
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False, use_legacy=False):
        cd = torch.load(path, map_location="cpu")
        if "state_dict" in list(cd.keys()):
            cd = cd["state_dict"]
        if use_legacy: cd = self.convert_legacy(cd)
        keys = list(cd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del cd[k]
        missing, unexpected = self.load_state_dict(cd, strict=False) if not only_model else self.model.load_state_dict(
            cd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
     
    def get_input(self, batch, data_key, cond_key):
        x = batch.get(data_key)
        x = rearrange(f.one_hot(x.long(), self.num_classes), "b 1 h w d x -> b x h w d")
        c = self.cond_stage_model(batch.get(cond_key))
        c = {f"c_{self.conditioning_key}": [c.float()]}
        ret = [x.float(), c]
        return ret
    
    def get_noise(self, x):
        if self.noise_type == "uniform":
            return torch.full_like(x, fill_value=1 / self.num_classes)
        elif self.noise_type == "categorical":
            return OneHotCategoricalBCHW(logits=torch.zeros_like(x)).sample()
    
    def lpips_loss(self, x0pred, x0):
        x, y = map(lambda i: repeat(i.argmax(1, keepdim=True), 'b 1 d h w -> b c d h w', c=3), [x0pred, x0])
        if self.dims == 3:
            lpips_x = self.lpips(rearrange(x, "b c d h w -> (b d) c h w"),
                                rearrange(y, "b c d h w -> (b d) c h w")).mean()
            lpips_y = self.lpips(rearrange(x, "b c d h w -> (b h) c d w"),
                                rearrange(y, "b c d h w -> (b h) c d w")).mean()
            lpips_z = self.lpips(rearrange(x, "b c d h w -> (b w) c d h"),
                                rearrange(y, "b c d h w -> (b w) c d h")).mean()
            lpips = (lpips_x + lpips_y + lpips_z) / 3
        elif self.dims == 2:
            lpips = self.lpips(x, y)
        return lpips
    
    def q_xt_given_xtm1(self, xtm1, t, noise=None):
        betas = extract_into_tensor(self.betas, t, xtm1.shape)
        if exists(noise): probs = (1 - betas) * xtm1 + betas * noise
        else: probs = (1 - betas) * xtm1 + betas / self.num_classes
        return probs

    def q_xt_given_x0(self, x0, t, noise=None):
        alphas_cumprod = extract_into_tensor(self.alphas_cumprod, t, x0.shape)
        if exists(noise): probs = alphas_cumprod * x0 + (1 - alphas_cumprod) * noise
        else: probs = alphas_cumprod * x0 + (1 - alphas_cumprod) / self.num_classes
        return probs

    def q_xtm1_given_x0_xt(self, xt, x0, t, noise=None):
        # computes q_xtm1 given q_xt, q_x0, noise
        alphas_t = extract_into_tensor(self.alphas, t, x0.shape)
        alphas_cumprod_tm1 = extract_into_tensor(self.alphas_cumprod_prev, t, x0.shape)
        if exists(noise): theta = ((alphas_t * xt + (1 - alphas_t) * noise) * (alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) * noise))
        else: theta = ((alphas_t * xt + (1 - alphas_t) / self.num_classes) * (alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) / self.num_classes))
        return theta / theta.sum(dim=1, keepdim=True)

    def q_xtm1_given_x0pred_xt(self, xt, theta_x0, t):
        # computes q_xtm1 given q_xt, q_x0, noise by evaluating every possible x0=l\in[1...L] and p(x0=l|xt)
        alphas_t = extract_into_tensor(self.alphas, t, xt.shape)
        alphas_cumprod_tm1 = extract_into_tensor(self.alphas_cumprod_prev, t, xt.shape)[..., None]

        # We need to evaluate theta_post for all values of x0
        x0 = torch.eye(self.num_classes, device=xt.device)[None, :, :, None, None]
        if self.dims == 3: x0 = x0[..., None]
        theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) / self.num_classes # [B, C, H, W]
        theta_xtm1_x0 = alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) / self.num_classes # [B, C1, C2, H, W]

        aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
        theta_xtm1_xtx0 = aux / aux.sum(dim=1, keepdim=True) # [B, C1, C2, H, W]
        return torch.einsum("bcd..., bd... -> bc...", theta_xtm1_xtx0, theta_x0)
    
    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.data_key, self.cond_key)
        loss = self(x, c, class_id=batch.get("class_id"))
        return loss
    
    def forward(self, x, c, *args, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.multinomial(torch.arange(self.timesteps, device=self.device) ** 1.5, x.shape[0])
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        return self.p_losses(x, c, t, *args, **kwargs)
    
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
    
    def p_losses(self, x0, c, t, class_id=None, *args, **kwargs):
        noise = self.get_noise(x0)
        q_xt = self.q_xt_given_x0(x0, t, noise if self.parameterization != "kl" else None)
        model_outputs = self.model(q_xt, t, **c)
        if isinstance(model_outputs, dict): 
            label_cond = model_outputs.get("label_out")
            model_outputs = model_outputs["diffusion_out"]
            
        loss_log = dict()
        log_prefix = 'train' if self.training else 'val'
        if self.parameterization == "kl":
            xt = OneHotCategoricalBCHW(q_xt).sample()
            q_xtm1_given_xt_x0 = self.q_xtm1_given_x0_xt(xt, x0, t,)
            q_xtm1_given_xt_x0pred = self.q_xtm1_given_x0pred_xt(xt, model_outputs, t,)
            
            kl_loss = f.kl_div(torch.log(torch.clamp(q_xtm1_given_xt_x0pred, min=1e-12)),
                                q_xtm1_given_xt_x0,
                                reduction='none')
            kl_loss_per_class = kl_loss.sum(1) * self.class_weights[x0.argmax(1)]
            if (kl_loss_per_class.sum() < -1e-3).any():
                print(f"negative KL divergence {kl_loss_per_class.sum()} encountered in loss")
            batch_loss = kl_loss_per_class.sum() / x0.shape[0]
            loss_log[f"{log_prefix}/kl_div"] = batch_loss.item() * self.loss_config["kl_div"].get("coeff", 1)
        
        elif self.parameterization in ['x0', 'eps']:
            target = x0 if self.parameterization == 'x0' else noise
            if self.loss_type == 'l1':
                dir_loss = (model_outputs - target).sum()
            elif self.loss_type == 'l2':
                dir_loss = f.mse_loss(model_outputs, target, reduction='none').sum()
            batch_loss = dir_loss
            loss_log[f"{log_prefix}/dir_loss"] = batch_loss.item() * self.loss_config["dir_loss"].get("coeff", 1)
            
        if hasattr(self.model.diffusion_model, "use_label_predictor") and self.model.diffusion_model.use_label_predictor:
            ce_loss = f.cross_entropy(label_cond, class_id)
            loss_log[f"{log_prefix}/ce"] = ce_loss
            batch_loss += ce_loss
        
        loss_log[f"{log_prefix}/loss"] = batch_loss.item()
        loss_log["debug"] = model_outputs.argmax(1).max()

        if not self.automatic_optimization and self.training:
            opt = self.optimizers()
            if self.use_scheduler:
                lr = opt.param_groups[0]['lr']
                self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
                
            opt.zero_grad()
            self.manual_backward(batch_loss)
            opt.step()
        
        return batch_loss, loss_log
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self,):
        if self.use_scheduler:
            sch = self.lr_schedulers()
            if isinstance(sch, ReduceLROnPlateau): sch.step(self.trainer.callback_metrics["loss"])
            else: sch.step()
        
        if self.use_ema:
            self.model_ema(self.model)
            
    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        logs = dict()
        logs1 = self._log_images(use_ddim=False, *args, **kwargs)
        for k, v in logs1.items():
            if k not in ["samples"]:
                logs[k] = v
        logs["samples"] = logs1["samples"]
        if not self.training and self.parameterization != "kl":
            logs2 = self._log_images(use_ddim=True, *args, **kwargs)
            logs["ddim_samples"] = logs2["samples"]
        return logs
    
    @torch.no_grad()
    def _log_images(self, batch, split="train", sample=True, use_ddim=False, ddim_steps=50, ddim_eta=1., verbose=False, **kwargs):
        x0, c = self.get_input(batch, self.data_key, self.cond_key)
        noise = self.get_noise(x0)
        
        logs = dict()
        b = x0.shape[0]
        logs["inputs"] = x0.argmax(1)
        logs["conditioning"] = str(batch["text"])
        if sample:
            if split == "train":
                t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
                q_xt = self.q_xt_given_x0(x0, t, noise if self.parameterization != "kl" else None)
                model_outputs = self.model(q_xt, t, **c)
                if isinstance(model_outputs, dict): 
                    model_outputs = model_outputs["diffusion_out"]
                x0pred = OneHotCategoricalBCHW(model_outputs).max_prob_sample()
                
                logs["t"] = str(t.cpu().numpy().tolist())
                logs["xt"] = OneHotCategoricalBCHW(q_xt).sample().argmax(1)
                logs["samples"] = x0pred.argmax(1)
                
                lpips = self.lpips_loss(x0pred.argmax(1, keepdim=True), x0.argmax(1, keepdim=True))
                self.log(f"{split}/lpips_metric", lpips, prog_bar=True, on_step=True)
                
            else:
                if not use_ddim: logs = logs | self.p_sample(noise, c, verbose=verbose, **kwargs)
                else: logs = logs | self.p_sample_ddim(noise, c, make_ddim_timesteps("uniform", 
                                                                                     num_ddim_timesteps=ddim_steps, 
                                                                                     num_ddpm_timesteps=self.timesteps,
                                                                                     verbose=verbose), verbose=verbose, **kwargs)
                x0pred = logs["samples"]

                lpips = self.lpips_loss(x0pred.unsqueeze(1), x0.argmax(1, keepdim=True))
                self.log(f"{split}/lpips_metric", lpips, prog_bar=True, on_step=True)
            
        return logs
    
    @torch.no_grad()
    def p_sample(self, q_xT=None, c=None, verbose=False, 
                 plot_progressive_rows=False, 
                 plot_denoising_rows=False, plot_diffusion_every_t=200):
        logs = dict()
        with self.ema_scope():
            c = c if exists(c) else dict()
            p_xt, b = q_xT, q_xT.shape[0]
            t_values = reversed(range(1, self.timesteps)) if not verbose else tqdm(reversed(range(1, self.timesteps)), total=self.timesteps - 1, desc="sampling progress")
            
            if plot_denoising_rows: denoising_rows = []
            if plot_progressive_rows: progressive_rows = []
            for t in t_values:
                t_ = torch.full(size=(b,), fill_value=t, device=q_xT.device)
                model_outputs = self.model(p_xt, t_, **c)
                if isinstance(model_outputs, dict): 
                    model_outputs = model_outputs["diffusion_out"]
                    
                if self.parameterization == "eps":
                    alphas_t = extract_into_tensor(self.alphas, t_, p_xt.shape)
                    p_x0_given_xt = (p_xt - (1 - alphas_t) * model_outputs) / alphas_t
                elif self.parameterization == "x0":
                    p_x0_given_xt = model_outputs
                
                if self.parameterization == "kl":
                    p_x0_given_xt = model_outputs
                    p_xt = torch.clamp(self.q_xtm1_given_x0pred_xt(p_xt, p_x0_given_xt, t_), min=1e-12)
                    p_xt = OneHotCategoricalBCHW(probs=p_xt).sample()
                else:
                    eps = self.get_noise(torch.zeros_like(q_xT))
                    p_xt = torch.clamp(self.q_xtm1_given_x0_xt(p_xt, p_x0_given_xt, t_, eps), min=1e-12)
                
                if plot_denoising_rows and t % plot_diffusion_every_t == 0: denoising_rows.append(p_x0_given_xt)
                if plot_progressive_rows and t % plot_diffusion_every_t == 0: progressive_rows.append(p_xt)

        if self.p_x1_sample == "majority":
            x0pred = OneHotCategoricalBCHW(probs=p_xt).max_prob_sample()
        elif self.p_x1_sample == "confidence":
            x0pred = OneHotCategoricalBCHW(probs=p_xt).prob_sample()
            
        logs["samples"] = x0pred.argmax(1)
        if plot_denoising_rows:
            denoising_rows = OneHotCategoricalBCHW(probs=torch.cat(denoising_rows, dim=0)).max_prob_sample()
            logs["p(x0|xt) at different timestep"] = denoising_rows.argmax(1)
        if plot_progressive_rows:
            progressive_rows = OneHotCategoricalBCHW(probs=torch.cat(progressive_rows, dim=0)).max_prob_sample()
            logs["p(x_{t-1}|xt) at different timestep"] = progressive_rows.argmax(1)
        
        return logs
            
    @torch.no_grad()
    def p_sample_ddim(self, q_xT, c, ddim_timesteps, verbose=False):
        logs = dict()
        
        def q_xtm1_given_x0_xt_ddim(xt, x0, ddim_t, ddim_tm1, noise=None):
            # computes q_xtm1 given q_xt, q_x0, noise
            alphas_t = extract_into_tensor(self.alphas, ddim_t, x0.shape)
            alphas_cumprod_tm1 = extract_into_tensor(self.alphas_cumprod_prev, ddim_tm1, x0.shape)
            if exists(noise): theta = ((alphas_t * xt + (1 - alphas_t) * noise) * (alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) * noise))
            else: theta = ((alphas_t * xt + (1 - alphas_t) / self.num_classes) * (alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) / self.num_classes))
            return theta / theta.sum(dim=1, keepdim=True)
        
        def q_xtm1_given_x0pred_xt_ddim(xt, x0pred, ddim_t, ddim_tm1, noise=None):
            alphas_t = extract_into_tensor(self.alphas, ddim_t, xt.shape)
            alphas_cumprod_tm1 = extract_into_tensor(self.alphas_cumprod_prev, ddim_tm1, xt.shape)
            
            x0 = torch.eye(self.num_classes, device=xt.device)[None, :, :, None, None]
            if self.dims == 3: x0 = x0[..., None]
            theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) / self.num_classes
            theta_xtm1_x0 = alphas_cumprod_tm1 * x0 + (1 - alphas_cumprod_tm1) / self.num_classes
            aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
            theta_xtm1_xtx0 = aux / aux.sum(dim=1, keepdim=True)
            return torch.einsum("bcd...,bd...->bc...", theta_xtm1_xtx0, x0pred)
        
        p_xt, b = q_xT, q_xT.shape[0]
        t_values = list(reversed(range(1, len(ddim_timesteps)))) 
        iterator = t_values if not verbose else tqdm(t_values, total=len(ddim_timesteps), desc="ddim sampling progress")
        for index, t in enumerate(iterator):
            t_ = torch.full(size=(b,), fill_value=ddim_timesteps[t], device=q_xT.device)
            t_m1 = torch.full(size=(b,), fill_value=ddim_timesteps[t_values[index + 1] if index + 1 < len(t_values) else 0], device=q_xT.device)
            
            model_outputs = self.model(p_xt, t_, **c)
            if isinstance(model_outputs, dict): 
                model_outputs = model_outputs["diffusion_out"]
                
            if self.parameterization == "eps":
                alphas_t = extract_into_tensor(self.alphas, t_, p_xt.shape)
                p_x0_given_xt = (p_xt - (1 - alphas_t) * model_outputs) / alphas_t
            elif self.parameterization == "x0":
                p_x0_given_xt = model_outputs
                
            if self.parameterization == "kl":
                p_x0_given_xt = model_outputs
                p_xt = torch.clamp(q_xtm1_given_x0pred_xt_ddim(p_xt, p_x0_given_xt, t_, t_m1), min=1e-12)
                p_xt = OneHotCategoricalBCHW(probs=p_xt).sample()
            else:
                eps = self.get_noise(torch.zeros_like(q_xT))
                p_xt = torch.clamp(q_xtm1_given_x0_xt_ddim(p_xt, p_x0_given_xt, t_, t_m1, eps), min=1e-12)
            
        if self.p_x1_sample == "majority":
            x0pred = OneHotCategoricalBCHW(probs=p_xt).max_prob_sample()
        elif self.p_x1_sample == "confidence":
            x0pred = OneHotCategoricalBCHW(probs=p_xt).prob_sample()
            
        logs["samples"] = x0pred.argmax(1)
        return logs
            
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_parameters = list(self.model.parameters())
        if self.cond_stage_trainable:
            opt_parameters = opt_parameters + list(self.cond_stage_model.parameters())
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
    
    
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


if __name__ == "__main__":
    spec = OmegaConf.to_container(OmegaConf.load("./run/train_ruijin_ccdm.yaml"))
            
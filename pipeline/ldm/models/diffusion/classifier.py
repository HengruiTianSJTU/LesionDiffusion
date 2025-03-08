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

__models__ = {
    'image_enc': EncoderUNetModel,
    'text_enc': TransformerEmbedder
}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class NoisyLatentImageClassifier(pl.LightningModule):

    def __init__(self,
                 diffusion_path,
                 num_classes,
                 ckpt_path=None,
                 pool='attention',
                 label_key=None,
                 diffusion_ckpt_path=None,
                 scheduler_config=None,
                 weight_decay=1.e-2,
                 log_steps=10,
                 monitor='val/loss',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        # get latest config of diffusion model
        diffusion_config = natsorted(glob(os.path.join(diffusion_path, 'configs', '*-project.yaml')))[-1]
        self.diffusion_config = OmegaConf.load(diffusion_config).model
        self.diffusion_config.params.ckpt_path = diffusion_ckpt_path
        self.load_diffusion()

        self.monitor = monitor
        self.numd = self.diffusion_model.first_stage_model.encoder.num_resolutions - 1
        self.log_time_interval = self.diffusion_model.num_timesteps // log_steps
        self.log_steps = log_steps

        self.label_key = label_key if not hasattr(self.diffusion_model, 'cond_stage_key') \
            else self.diffusion_model.cond_stage_key

        assert self.label_key is not None, 'label_key neither in diffusion model nor in model.params'

        if self.label_key not in __models__:
            raise NotImplementedError()

        self.load_classifier(ckpt_path, pool)

        self.scheduler_config = scheduler_config
        self.use_scheduler = self.scheduler_config is not None
        self.weight_decay = weight_decay

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

    def load_diffusion(self):
        model = instantiate_from_config(self.diffusion_config)
        self.diffusion_model = model.eval()
        self.diffusion_model.train = disabled_train
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

    def load_classifier(self, ckpt_path, pool):
        model_config = deepcopy(self.diffusion_config.params.unet_config.params)
        model_config.in_channels = self.diffusion_config.params.unet_config.params.out_channels
        model_config.out_channels = self.num_classes
        if self.label_key == 'class_label':
            model_config.pool = pool

        self.model = __models__[self.label_key](**model_config)
        if ckpt_path is not None:
            print('#####################################################################')
            print(f'load from ckpt "{ckpt_path}"')
            print('#####################################################################')
            self.init_from_ckpt(ckpt_path)

    @torch.no_grad()
    def get_x_noisy(self, x, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x))
        continuous_sqrt_alpha_cumprod = None
        if self.diffusion_model.use_continuous_noise:
            continuous_sqrt_alpha_cumprod = self.diffusion_model.sample_continuous_noise_level(x.shape[0], t + 1)
            # todo: make sure t+1 is correct here

        return self.diffusion_model.q_sample(x_start=x, t=t, noise=noise,
                                             continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod)

    def forward(self, x_noisy, t, *args, **kwargs):
        return self.model(x_noisy, t)

    @torch.no_grad()
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def get_conditioning(self, batch, k=None):
        if k is None:
            k = self.label_key
        assert k is not None, 'Needs to provide label key'

        targets = batch[k].to(self.device)

        if self.label_key == 'segmentation':
            targets = rearrange(targets, 'b h w c -> b c h w')
            for down in range(self.numd):
                h, w = targets.shape[-2:]
                targets = F.interpolate(targets, size=(h // 2, w // 2), mode='nearest')

            # targets = rearrange(targets,'b c h w -> b h w c')

        return targets

    def compute_top_k(self, logits, labels, k, reduction="mean"):
        _, top_ks = torch.topk(logits, k, dim=1)
        if reduction == "mean":
            return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
        elif reduction == "none":
            return (top_ks == labels[:, None]).float().sum(dim=-1)

    def on_train_epoch_start(self):
        # save some memory
        self.diffusion_model.model.to('cpu')

    @torch.no_grad()
    def write_logs(self, loss, logits, targets):
        log_prefix = 'train' if self.training else 'val'
        log = {}
        log[f"{log_prefix}/loss"] = loss.mean()
        log[f"{log_prefix}/acc@1"] = self.compute_top_k(
            logits, targets, k=1, reduction="mean"
        )
        log[f"{log_prefix}/acc@5"] = self.compute_top_k(
            logits, targets, k=5, reduction="mean"
        )

        self.log_dict(log, prog_bar=False, logger=True, on_step=self.training, on_epoch=True)
        self.log('loss', log[f"{log_prefix}/loss"], prog_bar=True, logger=False)
        self.log('global_step', self.global_step, logger=False, on_epoch=False, prog_bar=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, on_step=True, logger=True, on_epoch=False, prog_bar=True)

    def shared_step(self, batch, t=None):
        x, *_ = self.diffusion_model.get_input(batch, k=self.diffusion_model.first_stage_key)
        targets = self.get_conditioning(batch)
        if targets.dim() == 4:
            targets = targets.argmax(dim=1)
        if t is None:
            t = torch.randint(0, self.diffusion_model.num_timesteps, (x.shape[0],), device=self.device).long()
        else:
            t = torch.full(size=(x.shape[0],), fill_value=t, device=self.device).long()
        x_noisy = self.get_x_noisy(x, t)
        logits = self(x_noisy, t)

        loss = F.cross_entropy(logits, targets, reduction='none')

        self.write_logs(loss.detach(), logits.detach(), targets.detach())

        loss = loss.mean()
        return loss, logits, x_noisy, targets

    def training_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch)
        return loss

    def reset_noise_accs(self):
        self.noisy_acc = {t: {'acc@1': [], 'acc@5': []} for t in
                          range(0, self.diffusion_model.num_timesteps, self.diffusion_model.log_every_t)}

    def on_validation_start(self):
        self.reset_noise_accs()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch)

        for t in self.noisy_acc:
            _, logits, _, targets = self.shared_step(batch, t)
            self.noisy_acc[t]['acc@1'].append(self.compute_top_k(logits, targets, k=1, reduction='mean'))
            self.noisy_acc[t]['acc@5'].append(self.compute_top_k(logits, targets, k=5, reduction='mean'))

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [optimizer], scheduler

        return optimizer

    @torch.no_grad()
    def log_images(self, batch, N=8, *args, **kwargs):
        log = dict()
        x = self.get_input(batch, self.diffusion_model.first_stage_key)
        log['inputs'] = x

        y = self.get_conditioning(batch)

        if self.label_key == 'class_label':
            y = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
            log['labels'] = y

        if ismap(y):
            log['labels'] = self.diffusion_model.to_rgb(y)

            for step in range(self.log_steps):
                current_time = step * self.log_time_interval

                _, logits, x_noisy, _ = self.shared_step(batch, t=current_time)

                log[f'inputs@t{current_time}'] = x_noisy

                pred = F.one_hot(logits.argmax(dim=1), num_classes=self.num_classes)
                pred = rearrange(pred, 'b h w c -> b c h w')

                log[f'pred@t{current_time}'] = self.diffusion_model.to_rgb(pred)

        for key in log:
            log[key] = log[key][:N]

        return log
    
    
class CharacteristicClassifier(pl.LightningModule):
    def __init__(self, 
                 encoder_config,
                 feature_encoder_config=None,
                 num_feature_classes={"sex": 2},
                 ckpt_path=None,
                 data_key="image",
                 feature_key=["sex"],
                 dims=3,
                 ignore_keys=[],
                 training_encoder=False,
                 only_load_encoder=False,
                 monitor='val/loss',
                 embed_dim=64,
                 activation="relu",
                 ):
        super().__init__()
        self.dims = dims
        self.data_key = data_key
        self.monitor = monitor
        self.feature_key = OmegaConf.to_container(feature_key)
        self.encoder = instantiate_from_config(encoder_config)
        self.training_encoder = training_encoder
        if not training_encoder:
            for p in self.encoder.parameters():
                p.detach_().requires_grad_(False)
        
        network_out = round(np.prod(np.array(self.encoder.resolution)) / 2 ** (self.dims * (self.encoder.num_resolutions - 1)))
        self.classifiers = torch.nn.ModuleDict()
        conv_nd = getattr(torch.nn, f"Conv{self.dims}d", torch.nn.Identity)
        batchnorm_nd = getattr(torch.nn, f"BatchNorm{self.dims}d", torch.nn.Identity)
        if activation == "relu":
            activation = torch.nn.ReLU
        elif activation == "leakyrelu":
            activation = torch.nn.LeakyReLU
        for key in feature_key:
            classifier = torch.nn.ModuleList()
            feature_encoder = Encoder(**feature_encoder_config)
            network_out = round(np.prod(np.array(feature_encoder.resolution)) / 2 ** (self.dims * (feature_encoder.num_resolutions - 1))) * feature_encoder_config["z_channels"]
            classifier.append(feature_encoder)
            classifier.append(torch.nn.Linear(network_out, num_feature_classes[key]))
            self.classifiers[key] = classifier
        
        if ckpt_path is not None: self.init_from_ckpt(ckpt_path, ignore_keys, only_load_encoder)
        
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
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.encoder.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        
    @torch.no_grad()
    def get_input(self, batch, k):
        if isinstance(k, (str, int)):
            x = batch[k]
            if len(x.shape) != self.dims + 2:
                x = x[:, None]
            x = x.to(memory_format=torch.contiguous_format).float()
            return x
        elif isinstance(k, (list, tuple)):
            xs = {}
            for kk in k:
                x = batch[kk]
                xs[kk] = x.to(memory_format=torch.contiguous_format)
            return xs
    
    @property
    def dataset_connector(self):
        return self.trainer._data_connector.trainer.datamodule.datasets["train"]

    def shared_step(self, batch):
        image = self.get_input(batch, self.data_key)
        feature = self.get_input(batch, self.feature_key)
        model_outputs = self.encoder(image)
        
        loss_all = 0.
        loss_log = {}
        log_prefix = "train" if self.training else "val"
        for k, classifier in self.classifiers.items():
            feat = feature[k]
            is_valid = (feature[k][:, 0].sum() == 0) * 1.
            preds = classifier[0](model_outputs)
            preds = classifier[1](preds.view(image.shape[0], -1))
            loss = torch.nn.functional.cross_entropy(preds, feat)
            loss_all += loss * is_valid
            loss_log[f"{log_prefix}/loss_{k}"] = loss.item() * is_valid
            
        self.log(f"{log_prefix}/loss", loss_all, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('global_step', self.global_step, logger=False, on_epoch=False, prog_bar=True, on_step=True)
        self.log_dict(loss_log, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss_all
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)
    
    @torch.no_grad()
    def log_images(self, batch, N=8, *args, **kwargs):
        logs = {}
        image = self.get_input(batch, self.data_key)
        feature = self.get_input(batch, self.feature_key)
        model_outputs = self.encoder(image)
        
        feat_log = {}
        for k, classifier in self.classifiers.items():
            preds = classifier[0](model_outputs)
            preds = classifier[1](preds.view(image.shape[0], -1))
            feat_log[k] = preds
            
        logs["inputs"] = image
        logs["feature_gt"] = self.dataset_connector.rev_parse(feature)
        logs["feature_pred"] = self.dataset_connector.rev_parse(feat_log)
        return logs
    
    def configure_optimizers(self):
        param = list(self.classifiers.parameters())
        if self.training_encoder:
            param += list(self.encoder.parameters())
        optimizer = AdamW(param, lr=self.learning_rate)
        return optimizer
        
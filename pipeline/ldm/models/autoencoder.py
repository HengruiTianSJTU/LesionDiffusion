import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.ema import LitEma

from ldm.util import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

# taming.modules.vqvae.quantize.VectorQuantizer
class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta, dims=3):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.dims = dims

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, reduction='mean'):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, "b c ... -> b ... c").contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        if reduction == 'mean':
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        elif reduction == 'none':
            loss = (z_q.detach()-z)**2 + self.beta * (z_q - z.detach()) ** 2

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b ... c -> b c ...").contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = rearrange(z_q, "b ... c -> b c ...").contiguous()

        return z_q


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 l1_weight=0.5,
                 dims=3, is_conditional=False, cond_key=None, conditioning_key="concat"
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        if "params" in lossconfig:
            lossconfig["params"]["dims"] = dims
            lossconfig["params"]["n_classes"] = n_embed
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, dims=dims)
                                        # remap=remap,
                                        # sane_index_shape=sane_index_shape)
        self.dims = dims
        self.l1_weight = l1_weight
        self.conv_nd = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d
        self.quant_conv = self.conv_nd(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = self.conv_nd(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor
        self.is_conditional = is_conditional
        if is_conditional:
            self.cond_key = cond_key
            self.conditioning_key = conditioning_key
            assert self.cond_key is not None

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x, c_cat=None):
        h = self.encoder(x, {f"c_{self.conditioning_key}": c_cat} if c_cat is not None else None)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, c_cat=None):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, {f"c_{self.conditioning_key}": c_cat} if c_cat is not None else None)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, cond=None, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input, cond)
        dec = self.decode(quant, cond)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if self.dims == 2:
            if len(x.shape) == 3:
                x = x[:, None]
            x = x.to(memory_format=torch.contiguous_format).float()
        elif self.dims == 3:
            if len(x.shape) == 4:
                x = x[:, None]
            x = x.to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        if self.is_conditional: c_cat = self.get_input(batch, self.cond_key)
        else: c_cat = None
        xrec, qloss, ind = self(x, c_cat, return_pred_indices=True)
        # loss = F.smooth_l1_loss(x, xrec) * self.l1_weight

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        if self.is_conditional: c_cat = self.get_input(batch, self.cond_key)
        else: c_cat = None
        xrec, qloss, ind = self(x, c_cat, return_pred_indices=True)
        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # if version.parse(pl.__version__) >= version.parse('1.4.0'): <then exec next line>
        del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.loss.frame_discriminator.parameters())+
                                    list(self.loss.ct_discriminator.parameters()),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if self.is_conditional: conditions = self.get_input(batch, self.cond_key)
        else: conditions = None
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x, conditions)
        if x.shape[1] == 3:
            # colorize with random projection
            assert xrec.shape[1] == 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x, conditions)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
                
        if self.is_conditional:
            y = self.get_input(batch, self.cond_key)
            log["conditioning"] = y
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x, c_cat=None):
        cf = {f"c_{self.conditioning_key}": c_cat} if c_cat is not None else None
        h = self.encoder(x, cf)
        h = self.quant_conv(h)
        return h

    def decode(self, h, c_cat=None, force_not_quantize=False, split_quantize=False):
        # also go through quantization layer
        cf = {f"c_{self.conditioning_key}": c_cat} if c_cat is not None else None
        if not force_not_quantize:
            if split_quantize:
                batch_size, channels, height, width, depth = h.shape
                split_width = width // 2
                split_depth = depth // 2
                
                # split latent to four parts
                h1 = h[:, :, :, :split_width, :split_depth]
                h2 = h[:, :, :, :split_width, split_depth:]
                h3 = h[:, :, :, split_width:, :split_depth]
                h4 = h[:, :, :, split_width:, split_depth:]
                
                quant1, emb_loss1, info1 = self.quantize(h1)
                quant2, emb_loss2, info2 = self.quantize(h2)
                quant3, emb_loss3, info3 = self.quantize(h3)
                quant4, emb_loss4, info4 = self.quantize(h4)
                
                # concat quantized result
                quant = torch.cat([
                    torch.cat([quant1, quant2], dim=-1),
                    torch.cat([quant3, quant4], dim=-1)
                ], dim=-2)
            else:
                quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)

        
        if split_quantize:
            base_device = quant.device
            current_device = int(str(base_device).split(':')[-1])
            next_device = current_device + 1
            model_device = torch.device(f'cuda:{next_device}')
            
            self.decoder = self.decoder.to(model_device)
            quant = quant.to(model_device)
            cf = cf.to(model_device) if cf is torch.Tensor else None

            dec = self.decoder(quant, cf, use_checkpoint=False).to(base_device)
        else: dec = self.decoder(quant, cf)
        return dec


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None, dims=3, is_conditional=False, cond_key=None, conditioning_key="concat"
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig, dims=dims)
        self.decoder = Decoder(**ddconfig, dims=dims)
        if "params" in lossconfig:
            lossconfig["params"]["dims"] = dims
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.dims = dims
        self.conv_nd = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d
        self.quant_conv = self.conv_nd(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = self.conv_nd(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.is_conditional = is_conditional
        if is_conditional:
            self.cond_key = cond_key
            self.conditioning_key = conditioning_key
            assert self.cond_key is not None

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, c_cat=None):
        h = self.encoder(x, {f"c_{self.conditioning_key}": c_cat} if c_cat is not None else None)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, c_cat=None):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, {f"c_{self.conditioning_key}": c_cat} if c_cat is not None else None)
        return dec

    def forward(self, input, conditions=None, sample_posterior=True):
        posterior = self.encode(input, conditions)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, conditions)
        return dec, posterior
    
    def reverse_forward(self, input, sample_posterior=True):
        pass

    def get_input(self, batch, k):
        x = batch[k]
        if self.dims == 2:
            if len(x.shape) == 3:
                x = x[:, None]
            x = x.to(memory_format=torch.contiguous_format).float()
        elif self.dims == 3:
            if len(x.shape) == 4:
                x = x[:, None]
            x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        if self.is_conditional: c_cat = self.get_input(batch, self.cond_key)
        else: c_cat = None
        reconstructions, posterior = self(inputs, c_cat)
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        if self.is_conditional: conditions = self.get_input(batch, self.cond_key)
        else: conditions = None
        reconstructions, posterior = self(inputs, conditions)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.loss.frame_discriminator.parameters()) + list(self.loss.ct_discriminator.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        if self.is_conditional: conditions = self.get_input(batch, self.cond_key)
        else: conditions = None
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x, conditions)
            if x.shape[1] == 3:
                # colorize with random projection
                assert xrec.shape[1] == 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()), conditions)
            log["reconstructions"] = xrec
        log["inputs"] = x
        if self.is_conditional:
            y = self.get_input(batch, self.cond_key)
            log["conditioning"] = y
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
    
    
class DoubleCodebookVQModel(VQModel):
    def __init__(self, **kw):
        n_embed = kw.get("n_embed", 2048)
        embed_dim = kw.get("embed_dim", 8)
        dims = kw.get("dims", 3)
        self.foreground_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, dims=dims)
        
    def encode(self, x, c_cat=None):
        h = self.encoder(x, {f"c_{self.conditioning_key}": c_cat} if c_cat is not None else None)
        h = self.quant_conv(h)
        quant_b, emb_loss_b, info_b = self.quantize(h, reduction="none")
        quant_f, emb_loss_f, info_f = self.foreground_quantize(h, reduction="none")
        
        return quant, emb_loss, info
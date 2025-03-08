import os, random, re, copy
import json
import h5py
import torch
import omegaconf
import torch.distributed
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndm

from enum import IntEnum
from tqdm import tqdm
from medpy.metric import binary
from einops import rearrange, repeat
from ldm.data.utils import load_or_write_split
from ldm.data.data_process_func import img_multi_thresh_normalized
from ldm.data.Torchio_contrast_dataloader import totalseg_class, save_to_nnunet
from ldm.util import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL, VQModel
from ldm.models.diffusion.ddpm import LatentDiffusion, InpaintingDiffusion, MaskDiffusion, InpaintingDiffusion_v2
from ldm.models.diffusion.ccdm import CategoricalDiffusion, OneHotCategoricalBCHW
from ldm.models.diffusion.classifier import CharacteristicClassifier
from ldm.models.downstream.efficient_subclass import EfficientSubclassSegmentation

from ldm.models.diffusion.ddim import make_ddim_timesteps
from ldm.modules.diffusionmodules.util import extract_into_tensor


def exists(x):
    return x is not None


class MetricType(IntEnum):
    lpips = 1
    fid = 2
    psnr = 3
    fvd = 4
    
    
class ComputeMetrics:
    def __init__(self, eval_scheme, fvd_config=None):
        self.eval_scheme = eval_scheme
        if MetricType.lpips in self.eval_scheme:
            from ldm.modules.losses.lpips import LPIPS
            self.perceptual = LPIPS().eval()
        if MetricType.fid in self.eval_scheme:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
            # self.fid.set_dtype(torch.float64)
        if MetricType.psnr in self.eval_scheme:
            from torchmetrics.image.psnr import PeakSignalNoiseRatio
            self.psnr = PeakSignalNoiseRatio()
        if MetricType.fvd in self.eval_scheme:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fvd_module: nn.Module = instantiate_from_config(fvd_config)
            self.fvd = FrechetInceptionDistance(feature=self.fvd_module, normalize=True)
    
    @torch.no_grad()
    def log_eval(self, x, y, log_group_metrics_in_2d=False):
        metrics = dict()
        assert x.shape == y.shape
        b, c, *shp = x.shape
        
        if MetricType.lpips in self.eval_scheme:
            # lower = better (0, ?)
            perceptual = self.perceptual
            if c != 1: x, y = map(lambda i: i[:, 0:1], [x, y])
            x, y = map(lambda i: repeat(i, 'b 1 ... -> b c ...', c=3), [x, y])
            if len(shp) == 3:
                lpips_x = perceptual(rearrange(x, "b c d h w -> (b d) c h w"),
                                    rearrange(y, "b c d h w -> (b d) c h w")).mean()
                lpips_y = perceptual(rearrange(x, "b c d h w -> (b h) c d w"),
                                    rearrange(y, "b c d h w -> (b h) c d w")).mean()
                lpips_z = perceptual(rearrange(x, "b c d h w -> (b w) c d h"),
                                    rearrange(y, "b c d h w -> (b w) c d h")).mean()
                lpips = (lpips_x + lpips_y + lpips_z) / 3
            elif len(shp) == 2:
                lpips = perceptual(x, y)
                
            metrics["LPIPS"] = lpips.item()
            
        if log_group_metrics_in_2d:
            if MetricType.fid in self.eval_scheme:
                # lower = better (0, inf)
                assert len(shp) == 3
                x, y = map(lambda i: rearrange(i, "b c h ... -> (b h) 1 c ..."), [x, y])
                # resize x and y to (3, 299, 299)
                x, y = map(lambda x: torch.nn.functional.interpolate(x, (3, 299, 299)).squeeze(1), [x, y])
                self.fid.update(x.float(), real=False)
                self.fid.update(y.float(), real=True)
                fid = self.fid.compute()
            
                metrics["FID"] = fid.item()
                
            if MetricType.psnr in self.eval_scheme:
                # larger = better (0, inf)
                x = rearrange(x, "b c h ... -> (b h) c ...")
                y = rearrange(y, "b c h ... -> (b h) c ...")
                psnr = self.psnr(x, y)
            
                metrics["PSNR"] = psnr.item()
        return metrics

    @torch.no_grad()
    def log_eval_group(self, xs, ys, group_size=10, sample=False, sample_num=40):
        if isinstance(xs, str | os.PathLike):
            xs = [os.path.join(xs, p) for p in os.listdir(xs)]
        if isinstance(ys, str | os.PathLike):
            ys = [os.path.join(ys, p) for p in os.listdir(ys)]
        
        metrics = dict()
        ext = '.'.join(xs[0].split('.')[1:])
        if ext == "nii.gz": load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        elif ext == "npz": load_fn = lambda x: np.load(x)[np.load(x).files[0]]
        
        x_buffer, y_buffer, i_buffer_fills = [], [], 0
        for x, y in zip(xs, ys):
            if len(x_buffer) < group_size:
                x, y = map(lambda i: torch.tensor(load_fn(i)), [x, y])
                x_buffer.append(x)
                y_buffer.append(y)
            else:
                i_buffer_fills += 1
                x, y = torch.cat(x_buffer, dim=0), torch.cat(y_buffer, dim=0)
                xp, yp = map(lambda i: rearrange(i, "b c h w d -> (b h) c w d"), [x, y])
                if sample:
                    random_indices = torch.randperm(xp.shape[0])[:min(xp.shape[0], sample_num)]
                    xp, yp = map(lambda i: i[random_indices], [xp, yp])
                    
                if MetricType.fid in self.eval_scheme:
                    xp, yp = map(lambda i: torch.nn.functional.interpolate(
                        rearrange(i, "b c h w -> b 1 c h w"), (3, 299, 299)).squeeze())
                    self.fid.update(xp, real=False)
                    self.fid.update(yp, real=True)
                    fid = self.fid.compute()
                    metrics["FID"] = metrics.get("FID", 0) + fid.item()
                if MetricType.psnr in self.eval_scheme:
                    psnr = self.psnr(xp, yp)
                    metrics["PSNR"] = metrics.get("PSNR", 0) + psnr.item()
                    
        metrics = {k: v / i_buffer_fills for k, v in metrics.items()}
        return metrics
    
    
class MakeDataset:
    def __init__(self, 
                 dataset_base,
                 include_keys,
                 suffixes,
                 bs=1, create_split=False, dims=3, desc=None):
        self.bs = bs
        self.dims = dims
        self.desc = desc
        self.base = dataset_base
        self.suffixes = suffixes
        self.include_keys = include_keys
        self.create_split = create_split
        
        self.dataset = dict()
        for key in suffixes.keys(): os.makedirs(os.path.join(self.base, key), exist_ok=True)
        
    def add(self, samples, sample_names=None, dtypes={}):
        if not exists(sample_names): sample_names = [f"case_{len(self.dataset) + i}" for i in range(self.bs)]
        if isinstance(sample_names, str): sample_names = [sample_names]
        for i in range(len(sample_names)): 
            while sample_names[i] in self.dataset: sample_names[i] = sample_names[i] + "0"
            self.dataset[sample_names[i]] = {}
        
        for key in self.include_keys:
            value = samples[key]
            for b in range(len(samples[key])):
                value_b = value[b]
                sample_name_b = sample_names[b]
                if isinstance(value_b, torch.Tensor):
                    f = os.path.join(self.base, key, sample_name_b + self.suffixes[key])
                    im = value_b.cpu().data.numpy().astype(dtypes.get(key, np.float32))
                    assert im.ndim == self.dims + 1, f"desired ndim {self.dims} and actual ndim {im.shape} not match"
                    sitk.WriteImage(sitk.GetImageFromArray(rearrange(self.postprocess(im), "c ... -> ... c")), f)
                    self.dataset[sample_name_b][key] = f
                else:
                    self.dataset[sample_name_b][key] = value_b
                    
    def postprocess(self, sample):
        # c h w d
        return sample
        
    def finalize(self, dt=None, **kw):
        dataset = {} | kw
        collect_dt = self.dataset if dt is None else dt
        dataset["data"] = collect_dt
        dataset["desc"] = self.desc
        dataset["keys"] = omegaconf.OmegaConf.to_container(self.include_keys)
        dataset["length"] = len(collect_dt)
        dataset["format"] = {k: self.suffixes.get(k, "raw") for k, v in self.suffixes.items()}
        
        if self.create_split:
            keys = list(collect_dt.keys())
            load_or_write_split(self.base, force=True, 
                                train=keys[:round(len(keys)*.7)],
                                val=keys[round(len(keys)*.7):round(len(keys)*.8)],
                                test=keys[round(len(keys)*.8):],)
        with open(os.path.join(self.base, "dataset.json"), "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
    
class AlignDataset(MakeDataset):
    def __init__(self, dataset_base, filename, include_keys, suffixes, bs=1, create_split=False, dims=3, desc=None, task='mask'):
        super().__init__(dataset_base, include_keys, suffixes, bs, create_split, dims, desc)
        self.task = task
        self.filename = filename
    def change_filename(self, filename):
        self.filename = filename
    def get_unique_file_path(self, base_path, suffix):
        index = 0
        file_path = f"{base_path}_{index}{suffix}"
        while os.path.exists(file_path):
            index += 1
            file_path = f"{base_path}_{index}{suffix}"
        return file_path
    def add(self, samples, sample_names=None, dtypes={}):
        for key in self.include_keys:
            value = samples[key]

            for b in range(len(value)):
                image_path = samples['image_path'][b]
                sample_name_b = os.path.dirname(image_path)
                if 'crop' in samples and samples['crop']:
                    crop_info = samples['crop'][b]
                    full_size = crop_info['full_size']
                    crop_start = crop_info['crop_start']
                    crop_size = crop_info['crop_size']

                value_b = value[b]

                if isinstance(value_b, torch.Tensor):
                    if self.task == 'mask':
                        full_image = torch.zeros((value_b.shape[0],) + full_size, dtype=value_b.dtype)

                        w1, h1, d1 = crop_start
                        w2, h2, d2 = w1 + crop_size[0], h1 + crop_size[1], d1 + crop_size[2]
                        full_image[:, w1:w2, h1:h2, d1:d2] = value_b

                        im = full_image.cpu().data.numpy().astype(np.float32)
                        assert im.ndim == self.dims + 1, f"desired ndim {self.dims} and actual ndim {im.shape} not match"
                        im = np.flip(im, axis=2)
                        im = np.where(im > 0.5, 2, 0).astype(np.uint8)

                        os.makedirs(sample_name_b, exist_ok=True)
                        file_path = os.path.join(sample_name_b, str(key) + self.suffixes[key])
                        sitk.WriteImage(sitk.GetImageFromArray(rearrange(im, "c ... -> ... c")), file_path)
                    elif self.task == 'edit':                        
                        img_nifti_path = os.path.join(sample_name_b, 'img_health.nii.gz')
                        img_nifti = sitk.ReadImage(img_nifti_path)
                        img_array = sitk.GetArrayFromImage(img_nifti).astype(np.float32)
                        
                        full_image = torch.zeros((value_b.shape[0],) + full_size, dtype=value_b.dtype)

                        w1, h1, d1 = crop_start
                        w2, h2, d2 = w1 + crop_size[0], h1 + crop_size[1], d1 + crop_size[2]
                        full_image[:, w1:w2, h1:h2, d1:d2] = value_b

                        im = full_image.cpu().data.numpy().astype(np.float32)
                        assert im.ndim == self.dims + 1, f"desired ndim {self.dims} and actual ndim {im.shape} not match"
                        im = np.flip(im, axis=2)
                        
                        norm_lis = [-1000, -200, 200, 1000]
                        thresh_lis = [0, 0.2, 0.8, 1]
                        im = img_multi_thresh_normalized(im, thresh_lis=thresh_lis, norm_lis=norm_lis, data_type=np.float32)

                        h1, h2 = full_size[1] - h2, full_size[1] - h1
                        
                        mean_im = np.mean(im[0, w1:w2, h1:h2, d1:d2])
                        mean_img_array = np.mean(img_array[w1:w2, h1:h2, d1:d2])
                        diff = mean_im - mean_img_array
                        im[0, w1:w2, h1:h2, d1:d2] -= diff
                        img_array[w1:w2, h1:h2, d1:d2] = im[0][w1:w2, h1:h2, d1:d2]
                        
                        edge_smooth_radius = 2
                        crop_bbox_mask = np.zeros(img_array.shape, dtype=np.float32)
                        x = np.linspace(0, 1, edge_smooth_radius * 2)
                        for dim in range(3):
                            if dim == 0:
                                x1 = x[:, np.newaxis, np.newaxis]
                                if w1 > edge_smooth_radius:
                                    crop_bbox_mask[w1 - edge_smooth_radius:w1 + edge_smooth_radius, h1:h2, d1:d2] = \
                                        np.maximum(crop_bbox_mask[w1 - edge_smooth_radius:w1 + edge_smooth_radius, h1:h2, d1:d2], x1)
                                if w2 < full_size[0] - edge_smooth_radius:    
                                    crop_bbox_mask[w2 - edge_smooth_radius:w2 + edge_smooth_radius, h1:h2, d1:d2] = \
                                        np.maximum(crop_bbox_mask[w2 - edge_smooth_radius:w2 + edge_smooth_radius, h1:h2, d1:d2], x1[::-1])
                            elif dim == 1:
                                x2 = x[np.newaxis, :, np.newaxis]
                                if h1 > edge_smooth_radius:
                                    crop_bbox_mask[w1:w2, h1 - edge_smooth_radius:h1 + edge_smooth_radius, d1:d2] = \
                                        np.maximum(crop_bbox_mask[w1:w2, h1 - edge_smooth_radius:h1 + edge_smooth_radius, d1:d2], x2)
                                if h2 < full_size[1] - edge_smooth_radius:
                                    crop_bbox_mask[w1:w2, h2 - edge_smooth_radius:h2 + edge_smooth_radius, d1:d2] = \
                                        np.maximum(crop_bbox_mask[w1:w2, h2 - edge_smooth_radius:h2 + edge_smooth_radius, d1:d2], x2[::-1])
                            elif dim == 2:
                                x3 = x[np.newaxis, np.newaxis, :]
                                if d1 > edge_smooth_radius:
                                    crop_bbox_mask[w1:w2, h1:h2, d1 - edge_smooth_radius:d1 + edge_smooth_radius] = \
                                        np.maximum(crop_bbox_mask[w1:w2, h1:h2, d1 - edge_smooth_radius:d1 + edge_smooth_radius], x3)
                                if d2 < full_size[2] - edge_smooth_radius:
                                    crop_bbox_mask[w1:w2, h1:h2, d2 - edge_smooth_radius:d2 + edge_smooth_radius] = \
                                        np.maximum(crop_bbox_mask[w1:w2, h1:h2, d2 - edge_smooth_radius:d2 + edge_smooth_radius], x3[::-1])
                        smoothed_img_array = ndm.gaussian_filter(img_array, sigma=1)
                        img_array = smoothed_img_array * crop_bbox_mask + img_array * (1 - crop_bbox_mask)

                        edited_img_path = os.path.join(sample_name_b, 'inpainting.nii.gz')
                        sitk.WriteImage(sitk.GetImageFromArray(img_array), edited_img_path)
                if isinstance(value_b, np.ndarray):
                    im = np.flip(value_b, axis=2)

                    os.makedirs(sample_name_b, exist_ok=True)
                    if self.task == 'mask':
                        filename = self.filename

                    elif self.task == 'edit':
                        filename = self.filename
                        norm_lis = [-1000, -200, 200, 1000]
                        thresh_lis = [0, 0.2, 0.8, 1]
                        im = img_multi_thresh_normalized(im, thresh_lis=thresh_lis, norm_lis=norm_lis, data_type=np.float32)
                        # save_to_nnunet(rearrange(im, "c ... -> ... c"), sample_name_b)
                    base_path = os.path.join(sample_name_b, filename)
                    file_path = self.get_unique_file_path(base_path, self.suffixes[key])
                    sitk.WriteImage(sitk.GetImageFromArray(rearrange(im, "c ... -> ... c")), file_path)

        
class InferAutoencoderVQ(VQModel, ComputeMetrics, MakeDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["data", "text"],
                 suffix_keys={"data":".nii.gz",},
                 **autoencoder_kwargs):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, include_keys, suffix_keys)
        VQModel.__init__(self, **autoencoder_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
        self.eval()
        
    def on_test_start(self, *args):
        if MetricType.fid in self.eval_scheme:
            self.fid = self.fid.to(self.device)
        if MetricType.fvd in self.eval_scheme:
            self.fvd = self.fvd.to(self.device)
        # if MetricType.lpips in self.eval_scheme:
        #     self.perceptual = self.perceptual.to(self.device)
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()   
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        logs = super(InferAutoencoderVQ, self).log_images(batch, *args, **kwargs)
        x = logs["inputs"]
        x_recon = logs["reconstructions"]
        if self.save_dataset:
            self.add(logs, batch.get("casename"), dtypes={"image": np.uint8})
        
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(x_recon, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        return {}, logs

class InferLatentDiffusion(LatentDiffusion, ComputeMetrics, MakeDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["data", "text"],
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            MakeDataset.__init__(self, save_dataset_path, include_keys, suffix_keys)
        LatentDiffusion.__init__(self, **diffusion_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
        self.eval()
        
    def on_test_start(self, *args):
        if MetricType.fid in self.eval_scheme:
            self.fid = self.fid.to(self.device)
        if MetricType.fvd in self.eval_scheme:
            self.fvd = self.fvd.to(self.device)
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        logs = super(InferLatentDiffusion, self).log_images(batch, *args, **kwargs)
        
        x = logs["inputs"]
        xbatchs = logs["samples"]
        if self.save_dataset:
            self.add(logs, batch.get("casename"), dtypes={"image": np.uint8})
            
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(xbatchs, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        
        return None, logs
    
class InferInpaintingDiffusion(InpaintingDiffusion, ComputeMetrics, AlignDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 include_keys=["data", "text"],
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            AlignDataset.__init__(self, save_dataset_path, include_keys, suffix_keys, task='edit')
        InpaintingDiffusion.__init__(self, **diffusion_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
        self.eval()
        
    def on_test_start(self, *args):
        if MetricType.fid in self.eval_scheme:
            self.fid = self.fid.to(self.device)
        if MetricType.fvd in self.eval_scheme:
            self.fvd = self.fvd.to(self.device)
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        if 'max_mask_num' in batch:
            for key in batch: # currently only for bs is 1
                if isinstance(batch[key], (list, tuple, np.ndarray, torch.Tensor)):
                    batch[key] = batch[key][0]
            original_device = None
            for key in batch:
                if torch.is_tensor(batch[key]):
                    original_device = batch[key].device
                    batch[key] = batch[key].cpu()
            img_path = batch['image_path']['image_path']
            crop_size = batch['crop_size']['crop_size']
            label_np = batch['label'].numpy()
            img_array = batch['image'].numpy().astype(np.float32)
            norm_lis = [-1000, -200, 200, 1000]
            thresh_lis = [0, 0.2, 0.8, 1]
            img_array = img_multi_thresh_normalized(img_array, thresh_lis=thresh_lis, norm_lis=norm_lis, data_type=np.float32)
            img_origin = img_array
            _, w, h, d = label_np.shape
            mask = (label_np == 2)
            labeled_array, num_features = ndm.label(mask)
            sizes = ndm.sum(mask, labeled_array, range(num_features + 1))
            sorted_indices = np.argsort(sizes[1:])[::-1] + 1

            max_mask_num = batch['max_mask_num']['max_mask_num']
            top_labels = sorted_indices[:min(max_mask_num, num_features)]  # biggest max_mask_num lesions，but less than actual quantity
            top_labels = [label for label in top_labels if sizes[label] > 50]

            if not top_labels: top_labels = [sorted_indices[0]]

            for chosenlabel in top_labels[::-1]:
                this_batch = {}
                region_mask = (labeled_array == chosenlabel)
                positions = np.where(region_mask)

                min_w, max_w = positions[1].min(), positions[1].max()
                min_h, max_h = positions[2].min(), positions[2].max()
                min_d, max_d = positions[3].min(), positions[3].max()
                
                center_w = (min_w + max_w) // 2
                center_h = (min_h + max_h) // 2
                center_d = (min_d + max_d) // 2

                half_output_size = np.array(crop_size) // 2
                w1 = max(center_w - half_output_size[0], 0)
                h1 = max(center_h - half_output_size[1], 0)
                d1 = max(center_d - half_output_size[2], 0)

                w2 = min(w1 + crop_size[0], w)
                h2 = min(h1 + crop_size[1], h)
                d2 = min(d1 + crop_size[2], d)

                if (w2 - w1) < crop_size[0]:
                    if center_w - half_output_size[0] < 0:
                        w1 = 0
                        w2 = crop_size[0]
                    else:
                        w1 = w - crop_size[0]
                        w2 = w
                    center_w = (w1 + w2) // 2
                if (h2 - h1) < crop_size[1]:
                    if center_h - half_output_size[1] < 0:
                        h1 = 0
                        h2 = crop_size[1]
                    else: 
                        h1 = h - crop_size[1]
                        h2 = h
                    center_h = (h1 + h2) // 2
                if (d2 - d1) < crop_size[2]:
                    if center_d - half_output_size[2] < 0:
                        d1 = 0
                        d2 = crop_size[2]
                    else:
                        d1 = d - crop_size[2]
                        d2 = d
                    center_d = (d1 + d2) // 2
                    
                batch['label'] = torch.tensor(region_mask * 2)
                if chosenlabel != sorted_indices[0]:
                    batch['image'] = torch.tensor(img_array)
                for key in batch.keys():
                    if torch.is_tensor(batch[key]):
                        this_batch[key] = batch[key][:, w1:w2, h1:h2, d1:d2]
                    else: this_batch[key] = batch[key]
                
                if 'type' in batch:
                    json_dic = batch['type']
                    all_keys = list(json_dic.keys())
                    if len(all_keys) == 1:
                        selected_key = all_keys[0]
                    else:
                        if 'coarseg' in batch:
                            coarseg_labels = this_batch['coarseg'].unique().tolist()
                            existing_keys = [key for key in all_keys if int(key) in coarseg_labels]
                            if not existing_keys:
                                print(f'no pairing between type and coarseg in {img_path}')
                                existing_keys = all_keys
                            coarseg_tensor = this_batch['coarseg']
                            label_tensor = this_batch['label']
                            max_area = 0
                            for key in existing_keys:
                                key_int = int(key)
                                mask = (coarseg_tensor == key_int)
                                masked_label = label_tensor * mask
                                area = torch.sum(masked_label).item()
                                if area > max_area:
                                    max_area = area
                                    selected_key = key
                        else:
                            selected_key = random.choice(all_keys)
                            
                    json_dic_key = json_dic[selected_key][2]
                    json_dic_key["lesion location"] = totalseg_class[selected_key]
                    json_dic_key.pop("size", None)
                    match = re.search(r'approximately (-?\d+(\.\d+)?) HU', json_dic_key['CT value'])
                    if match:
                        ct_value = float(match.group(1))
                        json_dic_key['CT value'] = f'{ct_value} HU,'
                    this_batch['text'] = json_dic_key

                for key in this_batch.keys():
                    if torch.is_tensor(this_batch[key]):
                        this_batch[key] = this_batch[key][None].to(original_device)
                    else:
                        this_batch[key] = [this_batch[key]]
                this_batch['label'] = this_batch['label'].squeeze(1)
                        
                logs = super(InferInpaintingDiffusion, self).log_images(this_batch, *args, **kwargs)

                output_image = logs['samples'][0].cpu().numpy().astype(np.float32)
                output_image = img_multi_thresh_normalized(output_image, thresh_lis=thresh_lis, norm_lis=norm_lis, data_type=np.float32)
                mean_im = np.mean(output_image)
                mean_img_array = np.mean(img_origin[0, w1:w2, h1:h2, d1:d2])
                diff = mean_im - mean_img_array
                output_image -= diff

                img_array[0, w1:w2, h1:h2, d1:d2] = output_image

                edge_smooth_radius = 2
                crop_bbox_mask = np.zeros(img_array.shape, dtype=np.float32)[0]
                x = np.linspace(0, 1, edge_smooth_radius * 2)

                for dim in range(3):
                    if dim == 0:
                        x1 = x[:, np.newaxis, np.newaxis]
                        if w1 > edge_smooth_radius:
                            crop_bbox_mask[w1 - edge_smooth_radius:w1 + edge_smooth_radius, h1:h2, d1:d2] = \
                                np.maximum(crop_bbox_mask[w1 - edge_smooth_radius:w1 + edge_smooth_radius, h1:h2, d1:d2], x1)
                        if w2 < w - edge_smooth_radius:    
                            crop_bbox_mask[w2 - edge_smooth_radius:w2 + edge_smooth_radius, h1:h2, d1:d2] = \
                                np.maximum(crop_bbox_mask[w2 - edge_smooth_radius:w2 + edge_smooth_radius, h1:h2, d1:d2], x1[::-1])
                    elif dim == 1:
                        x2 = x[np.newaxis, :, np.newaxis]
                        if h1 > edge_smooth_radius:
                            crop_bbox_mask[w1:w2, h1 - edge_smooth_radius:h1 + edge_smooth_radius, d1:d2] = \
                                np.maximum(crop_bbox_mask[w1:w2, h1 - edge_smooth_radius:h1 + edge_smooth_radius, d1:d2], x2)
                        if h2 < h - edge_smooth_radius:
                            crop_bbox_mask[w1:w2, h2 - edge_smooth_radius:h2 + edge_smooth_radius, d1:d2] = \
                                np.maximum(crop_bbox_mask[w1:w2, h2 - edge_smooth_radius:h2 + edge_smooth_radius, d1:d2], x2[::-1])
                    elif dim == 2:
                        x3 = x[np.newaxis, np.newaxis, :]
                        if d1 > edge_smooth_radius:
                            crop_bbox_mask[w1:w2, h1:h2, d1 - edge_smooth_radius:d1 + edge_smooth_radius] = \
                                np.maximum(crop_bbox_mask[w1:w2, h1:h2, d1 - edge_smooth_radius:d1 + edge_smooth_radius], x3)
                        if d2 < d - edge_smooth_radius:
                            crop_bbox_mask[w1:w2, h1:h2, d2 - edge_smooth_radius:d2 + edge_smooth_radius] = \
                                np.maximum(crop_bbox_mask[w1:w2, h1:h2, d2 - edge_smooth_radius:d2 + edge_smooth_radius], x3[::-1])

                # gaussian blur
                crop_bbox_mask = crop_bbox_mask[None]
                smoothed_img_array = ndm.gaussian_filter(img_array, sigma=1)
                img_array = smoothed_img_array * crop_bbox_mask + img_array * (1 - crop_bbox_mask)
            logs['samples'] = img_array[None]

        else:
            logs = super(InferInpaintingDiffusion, self).log_images(batch, *args, **kwargs)

        
        x = logs["inputs"]
        xbatchs = logs["samples"]
        if self.save_dataset:
            self.add(logs, batch.get("casename"), dtypes={"image": np.uint8})
            
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(xbatchs, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        
        return None, logs

class InferInpaintingDiffusion_v2(InpaintingDiffusion_v2, ComputeMetrics, AlignDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 save_name='inpainting',
                 include_keys=["data", "text"],
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            AlignDataset.__init__(self, save_dataset_path, save_name, include_keys, suffix_keys, task='edit')
        InpaintingDiffusion_v2.__init__(self, **diffusion_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
        self.eval()
        
    def on_test_start(self, *args):
        if MetricType.fid in self.eval_scheme:
            self.fid = self.fid.to(self.device)
        if MetricType.fvd in self.eval_scheme:
            self.fvd = self.fvd.to(self.device)
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        logs = super(InferInpaintingDiffusion_v2, self).log_images(batch, *args, **kwargs)
        
        logs["inputs"] = logs["inputs"].type(torch.float32)
        logs["samples"] = logs["samples"].type(torch.float32)
        x = logs["inputs"]
        xbatchs = logs["samples"]
        logs['samples'] = logs['samples'].to('cpu').numpy().astype(np.float32)
        if self.save_dataset:
            self.add(logs, batch.get("casename"), dtypes={"image": np.uint8})
        logs['samples'] = xbatchs
            
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(xbatchs, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
                    
        return None, logs


class InferMaskDiffusion(MaskDiffusion, ComputeMetrics, AlignDataset):
    def __init__(self, 
                 eval_scheme=[1],
                 save_dataset=False,
                 save_dataset_path=None,
                 save_name='samples1',
                 include_keys=["data", "text"],
                 suffix_keys={"data":".nii.gz",},
                 **diffusion_kwargs):
        if save_dataset:
            self.save_dataset = save_dataset
            assert exists(save_dataset_path)
            AlignDataset.__init__(self, save_dataset_path, save_name, include_keys, suffix_keys, task='mask')
        MaskDiffusion.__init__(self, **diffusion_kwargs)
        ComputeMetrics.__init__(self, eval_scheme)
        self.eval()
        
    def on_test_start(self, *args):
        if MetricType.fid in self.eval_scheme:
            self.fid = self.fid.to(self.device)
        if MetricType.fvd in self.eval_scheme:
            self.fvd = self.fvd.to(self.device)
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    @torch.no_grad()
    def log_images(self, batch, log_metrics=False, log_group_metrics_in_2d=False, *args, **kwargs):
        if 'max_mask_num' in batch:
            for key in batch: # currently only for bs is 1
                if isinstance(batch[key], (list, tuple, np.ndarray, torch.Tensor)):
                    batch[key] = batch[key][0]
            original_device = None
            for key in batch:
                if torch.is_tensor(batch[key]):
                    original_device = batch[key].device
                    batch[key] = batch[key].cpu()
            img_path = batch['image_path']['image_path']
            parts = img_path.split('/')
            organ = parts[-4].lower()
            disease = parts[-3].lower()
            crop_size = batch['crop_size']['crop_size']
            label_np = batch['label'].numpy()
            _, w, h, d = label_np.shape
            final_result = np.zeros_like(label_np, dtype=np.uint8)
            organ_bg = torch.zeros_like(batch['label']).long()
            mask = (label_np == 2)
            labeled_array, num_features = ndm.label(mask)
            sizes = ndm.sum(mask, labeled_array, range(num_features + 1))
            sorted_indices = np.argsort(sizes[1:])[::-1] + 1

            max_mask_num = batch['max_mask_num']['max_mask_num']
            top_labels = sorted_indices[:min(max_mask_num, num_features)]  # biggest max_mask_num lesions，but less than actual quantity
            top_labels = [label for label in top_labels if sizes[label] > 50]

            if not top_labels: top_labels = [sorted_indices[0]]
            max_attempts = 2

            for chosenlabel in top_labels:
                this_batch = {}
                region_mask = (labeled_array == chosenlabel)
                positions = np.where(region_mask)

                min_w, max_w = positions[1].min(), positions[1].max()
                min_h, max_h = positions[2].min(), positions[2].max()
                min_d, max_d = positions[3].min(), positions[3].max()
                
                center_w = (min_w + max_w) // 2
                center_h = (min_h + max_h) // 2
                center_d = (min_d + max_d) // 2

                half_output_size = np.array(crop_size) // 2
                w1 = max(center_w - half_output_size[0], 0)
                h1 = max(center_h - half_output_size[1], 0)
                d1 = max(center_d - half_output_size[2], 0)

                w2 = min(w1 + crop_size[0], w)
                h2 = min(h1 + crop_size[1], h)
                d2 = min(d1 + crop_size[2], d)

                if (w2 - w1) < crop_size[0]:
                    if center_w - half_output_size[0] < 0:
                        w1 = 0
                        w2 = crop_size[0]
                    else:
                        w1 = w - crop_size[0]
                        w2 = w
                    center_w = (w1 + w2) // 2
                if (h2 - h1) < crop_size[1]:
                    if center_h - half_output_size[1] < 0:
                        h1 = 0
                        h2 = crop_size[1]
                    else: 
                        h1 = h - crop_size[1]
                        h2 = h
                    center_h = (h1 + h2) // 2
                if (d2 - d1) < crop_size[2]:
                    if center_d - half_output_size[2] < 0:
                        d1 = 0
                        d2 = crop_size[2]
                    else:
                        d1 = d - crop_size[2]
                        d2 = d
                    center_d = (d1 + d2) // 2
                                                                            
                new_label = np.zeros_like(label_np, dtype=np.uint8)
                new_label[:, min_w:max_w+1, min_h:max_h+1, min_d:max_d+1] = 1
                batch['label'] = torch.from_numpy(new_label).long()
                
                for key in batch.keys():
                    if torch.is_tensor(batch[key]):
                        this_batch[key] = batch[key][:, w1:w2, h1:h2, d1:d2]
                    else: this_batch[key] = batch[key]
                
                if 'type' in batch:
                    json_dic = batch['type']
                    all_keys = list(json_dic.keys())
                    if len(all_keys) == 1:
                        selected_key = all_keys[0]
                    else:
                        if 'coarseg' in batch:
                            coarseg_labels = this_batch['coarseg'].unique().tolist()
                            existing_keys = [key for key in all_keys if int(key) in coarseg_labels]
                            if not existing_keys:
                                print(f'no pairing between type and coarseg in {img_path}')
                                existing_keys = all_keys
                            coarseg_tensor = this_batch['coarseg']
                            label_tensor = torch.tensor(region_mask[:, w1:w2, h1:h2, d1:d2])
                            max_area = 0
                            for key in existing_keys:
                                key_int = int(key)
                                mask = (coarseg_tensor == key_int)
                                masked_label = label_tensor * mask
                                area = torch.sum(masked_label).item()
                                if area > max_area:
                                    max_area = area
                                    selected_key = key
                        else:
                            selected_key = random.choice(all_keys)
                    
                    json_dic_key = json_dic[selected_key][2]
                    json_dic_key["lesion location"] = totalseg_class[selected_key]
                    json_dic_key.pop("size", None)
                    json_dic_key.pop("CT value", None)
                    if "specific features" in json_dic_key and json_dic_key["specific features"] == "": 
                        json_dic_key.pop("specific features")
                    json_dic_key.pop("density variations", None)#temporary
                    json_dic_key.pop("density", None)#temporary
                    this_batch['text'] = json_dic_key

                    if 'seg' in batch:
                        seg_tensor = this_batch['seg'].long()
                        organ_tensor = batch['seg'].long()
                        if 'lung' in organ or 'pneumon' in organ:
                            combined_seg = torch.zeros_like(seg_tensor)
                            for key in all_keys:
                                key_int = int(key)
                                combined_seg = combined_seg | (seg_tensor == key_int)
                                organ_bg = organ_bg | (organ_tensor == key_int)
                            this_batch['target'] = combined_seg.long()
                        else:
                            selected_label = int(selected_key)
                            combined_seg = (seg_tensor == selected_label)
                            organ_bg = organ_bg | (organ_tensor == selected_label)
                            this_batch['target'] = combined_seg.long()
                        
                        if 'coarseg' in batch:
                            organs_to_check = ['colon', 'bladder', 'esophagus', 'gall', 'stomach']
                            if any(organ_substring in organ for organ_substring in organs_to_check):
                                if 'stone' in disease:
                                    this_batch['text']['cavity'] = "within the lumen of a hollow organ,"
                                else:
                                    this_batch['text']['cavity'] = "wall thickening,"
                            else:
                                coarseg_tensor = this_batch['coarseg']
                                selected_label = int(selected_key)
                                coarseg_tensor = (coarseg_tensor == selected_label).long()
                                if torch.sum(coarseg_tensor) - torch.sum(this_batch['target']) > 100:
                                    this_batch['text']['cavity'] = "within the parenchymal organ,"
                                else:
                                    this_batch['text']['cavity'] = "protruding from the parenchymal organ,"
                            this_batch['title'] = {'title': str(organ) + ' ' + str(disease) + ', ' + this_batch['text']['cavity']}

                this_batch['seg'] = this_batch['seg'].float() / 104
                for key in this_batch.keys():
                    if torch.is_tensor(this_batch[key]):
                        this_batch[key] = this_batch[key][None].to(original_device)
                    else:
                        this_batch[key] = [this_batch[key]]
                                        
                ref_mask = 2 * region_mask[:, w1:w2, h1:h2, d1:d2]
                for attempt in range(max_attempts):
                    # Define the range of sigmoid thresholds and step size
                    sigmoid_thresholds = np.arange(0.9, 0.2, -0.1)

                    # Initialize the best threshold and minimum difference
                    best_threshold = None
                    min_diff = float('inf')
                    best_binary_output_image = None
    
                    logs = super(InferMaskDiffusion, self).log_images(this_batch, *args, **kwargs)

                    output_image = logs['samples'].cpu().numpy().astype(np.float32)
                    
                    # Iterate over each sigmoid threshold
                    for sigmoid_threshold in sigmoid_thresholds:
                        # Generate binary image based on the current threshold
                        binary_output_image = 2 * (output_image > sigmoid_threshold)[0]

                        # Label connected components and find the largest one
                        labeled_output, num_features = ndm.label(binary_output_image)
                        if num_features > 0:
                            sizes = ndm.sum(binary_output_image, labeled_output, range(num_features + 1))
                            max_label = np.argmax(sizes[1:]) + 1  # Background is not considered, so start from 1
                            binary_output_image = (labeled_output == max_label) * 2  # Keep only the largest connected component

                        # Calculate the size difference with the regionmask
                        current_diff = np.abs(np.sum(binary_output_image) - np.sum(ref_mask))

                        # Update the best threshold and binary image if the current difference is smaller
                        if current_diff < min_diff:
                            min_diff = current_diff
                            best_threshold = sigmoid_threshold
                            best_binary_output_image = binary_output_image.copy()  
                          
                    # Calculate overlap with the current final result
                    overlap = np.sum(final_result[:, w1:w2, h1:h2, d1:d2] * best_binary_output_image)
                    pixel_threshold = 50 if 'stone' in disease else 400
                    if np.sum(best_binary_output_image == 2) > pixel_threshold and not overlap:
                        # Update the final result only where binary_output_image is 1
                        final_result[:, w1:w2, h1:h2, d1:d2][best_binary_output_image == 2] = 2
                        break  # Exit loop if condition is met
                    elif attempt == max_attempts - 1 and not overlap:
                        # If all attempts have been made and there is no overlap, use the last attempt's result
                        final_result[:, w1:w2, h1:h2, d1:d2][best_binary_output_image == 2] = 2
                         
            organ_bg = organ_bg.numpy().astype(np.uint8)
            organ_bg[final_result == 2] = 2
            logs['samples'] = organ_bg[None]

        else:
            logs = super(InferMaskDiffusion, self).log_images(batch, *args, **kwargs)
            
        x = logs["inputs"]
        xbatchs = logs["samples"]
        if self.save_dataset:
            self.add(logs, batch.get("casename"), dtypes={"image": np.uint8})
        logs['samples'] = torch.tensor(logs['samples'])
            
        if self.eval_scheme is not None and len(self.eval_scheme) > 0 and log_metrics:
            metrics = self.log_eval(xbatchs, x, log_group_metrics_in_2d)
            # print(metrics)
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return metrics, logs
        
        return None, logs
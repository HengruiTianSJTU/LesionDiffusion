import pathlib as pb

import re
import cv2
import torch
import itertools
import numpy as np
from PIL import Image

import imageio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.cm import get_cmap
from torchvision.utils import make_grid
from einops import rearrange
from scipy.ndimage import sobel, distance_transform_edt
from collections import namedtuple
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler


OrganClass = namedtuple("OrganClass", ["label_name", "totalseg_id", "color"])
abd_organ_classes = [
    OrganClass("unlabeled", 0, (0, 0, 0)),
    OrganClass("spleen", 1, (0, 80, 100)),
    OrganClass("kidney_left", 2, (119, 11, 32)),
    OrganClass("kidney_right", 3, (119, 11, 32)),
    OrganClass("liver", 5, (250, 170, 30)),
    OrganClass("stomach", 6, (220, 220, 0)),
    OrganClass("pancreas", 10, (107, 142, 35)),
    OrganClass("small_bowel", 55, (255, 0, 0)),
    OrganClass("duodenum", 56, (70, 130, 180)),
    OrganClass("colon", 57, (0, 0, 255)),
    OrganClass("urinary_bladder", 104, (0, 255, 255)),
    OrganClass("colorectal_cancer", 255, (0, 255, 0))
]


def find_vacancy(path):
    path = pb.Path(path)
    d, f, s = path.parent, path.name, ".".join([""] + path.name.split(".")[1:])
    exist_files = list(_.name for _ in d.glob(f"*{s}"))
    file_num = list(int(([-1] + re.findall(r"\d+", _))[-1]) for _ in exist_files)
    fa = [i for i in range(1000) if i not in file_num]
    vacancy = d / (f.split(s)[0] + str(fa[0]) + s)
    print("found vacancy at ", f.split(s)[0] + str(fa[0]) + s)
    return vacancy
    

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


def minmax(val, minimum=None, maximum=None):
    if maximum is not None:
        val = min(val, maximum)
    if minimum is not None:
        val = max(val, minimum)
    return val


def combine_mask_and_im(x, overlay_coef=0.2, colors=None, n=11, mask_normalied=True):
    # b (im (no-display) msk) h w (d)
    if len(x.shape) == 5: ndim = 3
    if len(x.shape) == 4: ndim = 2
    def find_mask_boundaries_nd(im, mask, color):
        boundaries = torch.zeros_like(mask)
        for i in range(1, 12):
            m = (mask == i).numpy()
            sobel_x = sobel(m, axis=1, mode='constant')
            sobel_y = sobel(m, axis=2, mode='constant')
            if ndim == 3:
                sobel_z = sobel(m, axis=3, mode='constant')

            boundaries = torch.from_numpy((np.abs(sobel_x) + np.abs(sobel_y) + (0 if ndim == 2 else np.abs(sobel_z))) * i) * (boundaries == 0) + boundaries * (boundaries != 0)
        im = color[boundaries.long()] * (boundaries[..., None] > 0) + im * (boundaries[..., None] == 0)
        return im
    
    image = 255 * x[:, 0, ..., None].clamp(0, 1).repeat(*([1] * (ndim+1)), 3)
    mask = x[:, -1].round() * n if mask_normalied else mask[:, 1]
    cmap = get_cmap("viridis")
    colors = torch.tensor([(0, 0, 0)] + [cmap(i)[:-1] for i in np.arange(0.3, n) / n] if colors is None else colors, device=image.device)
    colored_mask = (colors[mask.long()] * (mask[..., None] > 0) + image * (mask[..., None] == 0))
    colored_im = colored_mask * overlay_coef + image * (1-overlay_coef)
    colored_im = rearrange(find_mask_boundaries_nd(colored_im, mask, colors), "b ... c -> b c ...")
    return colored_im


def combine_mask_and_im_v2(x, 
                           overlay_coef=.8, 
                           colors=None, n_mask=11, mask_normalized=False, 
                           num_images=8, 
                           backend="cv2"):
    # x: b 2 h w (d)
    x = x.cpu().data.numpy()
    cmap = get_cmap("viridis")
    colors = [cmap(i)[:-1] for i in np.arange(0.3, n_mask) / n_mask] if colors is None else colors
    image = np.expand_dims(x[:, 0], -1).repeat(3, -1)
    image = (image - image.min()) / (image.max() - image.min())
    mask = (x[:, 1] * (n_mask if mask_normalized else 1)).astype(np.uint8)
    contours = np.zeros(mask.shape + (3,))
    if backend == "cv2":
        h = mask.shape[1]
        if x.ndim == 5: mask = rearrange(mask, "b h w d -> (b h) w d")
        for ib, b in enumerate(mask):
            for i in np.unique(b).flatten():
                if i != 0:
                    binary = (b == i).astype(np.uint8)
                    contour, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contours[ib // h, ib % h] if x.ndim == 5 else contours[ib], contour, -1, colors[i], 1)
                    cv2.destroyAllWindows()
    elif backend == "edt":
        for ib, b in enumerate(mask):
            for i in np.unique(b).flatten():
                if i != 0:
                    outline = 5
                    binary = (b == i).astype(np.uint8)
                    x1, x2 = np.where(np.any(binary, (1, 2)))[0][[0, -1]]
                    y1, y2 = np.where(np.any(binary, (0, 2)))[0][[0, -1]]
                    z1, z2 = np.where(np.any(binary, (0, 1)))[0][[0, -1]]
                    box = [slice(max(0, x1 - outline), min(binary.shape[0] - 1, x2 + outline)),
                            slice(max(0, y1 - outline), min(binary.shape[1] - 1, y2 + outline)),
                            slice(max(0, z1 - outline), min(binary.shape[2] - 1, z2 + outline))]
                    box_binary = binary[box[0], box[1], box[2]]
                    contour = distance_transform_edt(box_binary == 0, )
                    contour = (contour > 0) & (contour < 1.8)
                    contours[ib, box[0], box[1], box[2]][contour] = np.expand_dims(np.array(colors[i]), 0).repeat(contour.sum(), 0)
    contours[contours[..., 0] == 0, :] = image[contours[..., 0] == 0]
    colored_image = image * (1 - overlay_coef) + contours * overlay_coef
    
    b, h = colored_image.shape[:2]
    if h > num_images: colored_image = colored_image[:, ::h // num_images]
    colored_image = rearrange(colored_image, "b h w d c -> (b h) c w d")
    colored_image = make_grid(torch.tensor(colored_image), nrow=min(num_images, h), normalize=False, pad_value=255, padding=3)
    return colored_image.squeeze().data.cpu().numpy()
        

def visualize(image: torch.Tensor, n_mask: int=20, num_images=8, is_mask=False):
    is_mask = is_mask or image.dtype == torch.long
    if len(image.shape) == 5:
        image = image[:, 0] 
    if len(image.shape) == 4:
        b, h = image.shape[:2]
        if h > num_images: image = image[:, ::h // num_images]
        image = rearrange(image, "b h w d -> (b h) 1 w d")
    else: return image.squeeze().data.cpu().numpy()
    image = make_grid(image, nrow=min(num_images, h), normalize=not is_mask, pad_value=n_mask if is_mask else 1, padding=3)

    if is_mask:
        cmap = get_cmap("viridis")
        rgb = torch.tensor([(0, 0, 0)] + [cmap(i)[:-1] for i in np.arange(0.3, n_mask) / n_mask], device=image.device)
        colored_mask = rearrange(rgb[image.long()][0], "i j n -> 1 n i j")
        return colored_mask.squeeze().data.cpu().numpy()
    else:
        return image.squeeze().data.cpu().numpy()
    

def make_gif(image: torch.Tensor, path: str, n_mask: int=20, num_images=-1, is_mask=False):
    is_mask = is_mask or image.dtype == torch.long
    image = image.data.cpu().numpy()
    if len(image.shape) == 5:
        image = image[:, 0] 
    if len(image.shape) == 4:
        b, h = image.shape[:2]
        hs = np.random.choice(max(1, h - num_images)).astype(int)
        if num_images > 0 and h > num_images: image = image[:, hs: hs + num_images]
        image = rearrange(image, "b h w d -> h (b w) d")

    if is_mask:
        cmap = get_cmap("viridis")
        rgb = np.array([(0, 0, 0)] + [cmap(i)[:-1] for i in np.arange(0.3, n_mask) / n_mask], device=image.device)
        raw = (rgb[image.long()][0] * 255).astype(np.uint8)
    else:
        raw = (np.repeat(image[..., None], 3, -1) * 255).astype(np.uint8)
        
    # h (b w) d 3 as frames, width, height, channels
    imageio.mimsave(path, raw.tolist(), duration=.3)


def image_logger(dict_of_images, path, log_separate=False, **kwargs):
    ind_vis = {}
    for k, v in dict_of_images.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 4: ind_vis[str(k)] = kwargs.get(k, lambda x: visualize(x, is_mask=False))(v)
        elif isinstance(v, str): ind_vis[str(k)] = kwargs.get(k, lambda x: x)(v)
    h = max([getattr(x, "shape", [0, 0, 0])[1] for x in ind_vis.values()])
    w = sum([getattr(x, "shape", [0, 0, 0])[2] for x in ind_vis.values()])
    if not log_separate:
        fig = plt.figure(figsize=(minmax(w // 1024, 15, 30), minmax(h // 1024, 15, 20)), dpi=300)
        for i, (k, v) in enumerate(ind_vis.items()):
            ax = fig.add_subplot(len(dict_of_images), 1, i + 1)
            ax.set_title(k)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            if isinstance(v, np.ndarray):
                ax.imshow(rearrange(v, "c h w -> h w c"))
            if isinstance(v, str):
                linewidth = 100
                ax.set_facecolor("black")
                ax.imshow(np.zeros((5, 20)))
                ax.text(.2, 2.5, "\n".join([v[i * linewidth: (i + 1) * linewidth] for i in range(np.ceil(len(v) / linewidth).astype(int))]),
                        color="white",
                        size = 5)
                        # fontproperties=matplotlib.font_manager.FontProperties(size=7,
                        #                                                         fname='/ailab/user/dailinrui/data/dependency/Arial-Unicode-Bold.ttf'))
        
        plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
        image_from_plt = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plt = {"main": image_from_plt.reshape(fig.canvas.get_width_height()[::-1] + (3,))}
        plt.close(fig)
    else:
        image_from_plt = dict()
        assert callable(path)
        for i, (k, v) in enumerate(ind_vis.items()):
            fig = plt.figure(dpi=300)
            ax = fig.gca()
            ax.set_title(k)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            if isinstance(v, np.ndarray):
                ax.imshow(rearrange(v, "c h w -> h w c"))
            if isinstance(v, str):
                linewidth = 100
                ax.set_facecolor("black")
                ax.imshow(np.zeros((5, 20)))
                ax.text(.2, 2.5, "\n".join([v[i * linewidth: (i + 1) * linewidth] for i in range(np.ceil(len(v) / linewidth).astype(int))]),
                        color="white",
                        size = 5)
                        # fontproperties=matplotlib.font_manager.FontProperties(size=5,
                        #                                                         fname='/ailab/user/dailinrui/data/dependency/Arial-Unicode-Bold.ttf'))
            plt.savefig(path(k), dpi=300, bbox_inches="tight", pad_inches=0)
            image_from_plt_step = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plt[k] = image_from_plt_step.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
    return image_from_plt


class DistributedTwoStreamBatchSampler(DistributedSampler):
    def __init__(self, dataset, primary_indices, secondary_indices, batch_size, secondary_batch_size,
                 num_replicas=None, rank=None, shuffle=True,
                 seed=0, drop_last: bool = False) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.batch_sampler = TwoStreamBatchSampler(primary_indices=primary_indices,
                                                   secondary_indices=secondary_indices,
                                                   batch_size=batch_size,
                                                   secondary_batch_size=secondary_batch_size)
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(self.batch_sampler)

    def __len__(self) -> int:
        return len(self.batch_sampler)


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, iterate_on_primary_indices=False, **kwargs):
        self.batch_size = batch_size
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.iterate_on_primary_indices = iterate_on_primary_indices

        assert len(self.primary_indices) >= self.primary_batch_size > 0,\
            f"condition {len(self.primary_indices)} >= {self.primary_batch_size} > 0 is not satisfied"
        if len(self.secondary_indices) < self.secondary_batch_size:
            self.secondary_indices = self.secondary_indices + self.primary_indices
            print("using coarse labels extracted from fine labels as supervision")
        # assert len(self.secondary_indices) >= self.secondary_batch_size >= 0,\
        #     f"condition {len(self.secondary_indices)} >= {self.secondary_batch_size} >= 0 is not satisfied"

    def __iter__(self):
        # primary_iter = iterate_eternally(self.primary_indices, len(self.secondary_indices) // len(self.primary_indices) if not self.iterate_on_primary_indices else 1)
        if self.secondary_batch_size != 0:
            primary_iter = iterate_once(self.primary_indices) if self.iterate_on_primary_indices else iterate_eternally(self.primary_indices)
            secondary_iter = iterate_eternally(self.secondary_indices) if not self.iterate_on_primary_indices else iterate_once(self.secondary_indices)
            return (
                primary_batch + secondary_batch
                for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
            )
        else:
            primary_iter = iterate_once(self.primary_indices)
            return (primary_batch for primary_batch in grouper(primary_iter, self.primary_batch_size))
        
    def __len__(self):
        if self.iterate_on_primary_indices: return len(self.primary_indices) // self.primary_batch_size
        return len(self.secondary_indices) // self.secondary_batch_size
    
    
class DistributedTwoStreamBatchSampler(DistributedSampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, iterate_on_primary_indices=False, **kwargs):
        self.batch_size = batch_size
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.iterate_on_primary_indices = iterate_on_primary_indices

        assert len(self.primary_indices) >= self.primary_batch_size > 0,\
            f"condition {len(self.primary_indices)} >= {self.primary_batch_size} > 0 is not satisfied"
        if len(self.secondary_indices) < self.secondary_batch_size:
            self.secondary_indices = self.secondary_indices + self.primary_indices
            print("using coarse labels extracted from fine labels as supervision")
        # assert len(self.secondary_indices) >= self.secondary_batch_size >= 0,\
        #     f"condition {len(self.secondary_indices)} >= {self.secondary_batch_size} >= 0 is not satisfied"

    def __iter__(self):
        # primary_iter = iterate_eternally(self.primary_indices, len(self.secondary_indices) // len(self.primary_indices) if not self.iterate_on_primary_indices else 1)
        if self.secondary_batch_size != 0:
            primary_iter = iterate_once(self.primary_indices) if self.iterate_on_primary_indices else iterate_eternally(self.primary_indices)
            secondary_iter = iterate_eternally(self.secondary_indices) if not self.iterate_on_primary_indices else iterate_once(self.secondary_indices)
            return (
                primary_batch + secondary_batch
                for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
            )
        else:
            primary_iter = iterate_once(self.primary_indices)
            return (primary_batch for primary_batch in grouper(primary_iter, self.primary_batch_size))
        
    def __len__(self):
        if self.iterate_on_primary_indices: return len(self.primary_indices) // self.primary_batch_size 
        return len(self.secondary_indices) // self.secondary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
    
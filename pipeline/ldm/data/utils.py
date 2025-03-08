import os, os.path as path, pathlib as pb
import json, torchio as tio, shutil, nibabel as nib
import re, scipy.ndimage as ndimage, numpy as np

import torch

from typing import List
from datetime import datetime
from functools import reduce
from collections import defaultdict, namedtuple
from collections.abc import MutableMapping


OrganTypeBase = namedtuple("OrganTypeBase", ["name", "label"])
    

class TotalsegOrganType:
    def __init__(self, path):
        self.max_split = 0
        self.organtypes = dict()
        self.nested_organtypes = dict()
        with open(path) as f:
            for line in f.readlines():
                index, name, alias = line.split('\t')
                index = int(index)
                name_split = name.split('_')
                self.max_split = max(self.max_split, len(name_split))
                key = reduce(lambda x, y: x + y[0].upper() + y[1:], name_split)
                self.organtypes[key[0].upper() + key[1:]] = index
                
                nested_type = self.nested_organtypes
                while len(name_split) > 0:
                    name_split_pop = name_split.pop(0)
                    name_split_pop = name_split_pop[0].upper() + name_split_pop[1:]
                    if name_split_pop not in nested_type: 
                        nested_type[name_split_pop] = {} 
                        if len(name_split) == 0: nested_type[name_split_pop] = index
                    nested_type = nested_type[name_split_pop]
                    
        self.organtypes["Background"] = 0
        self.nested_organtypes["Background"] = 0
        
        _flatten_dict = dict()
        def _flatten_dict_values(d: MutableMapping, parent_key='') -> MutableMapping:
            items = []
            for k, v in d.items():
                new_key = parent_key + k if parent_key else k
                if isinstance(v, MutableMapping):
                    items.extend(_flatten_dict_values(v, new_key))
                else:
                    items.append(v)
            if parent_key: _flatten_dict[parent_key] = items
            return items

        _flatten_dict_values(self.nested_organtypes)
        self.organtypes.update(_flatten_dict)

TotalsegOrganTypeV1 = TotalsegOrganType("dependency/totalseg_v1_label_mapping.txt")
TotalsegOrganTypeV2 = TotalsegOrganType("dependency/totalseg_v2_label_mapping.txt")


class LabelParser:
    def __init__(self, totalseg_version="v1"):
        self.totalseg_version = totalseg_version
        self.totalseg_decoder = TotalsegOrganTypeV1 if totalseg_version == "v1" else TotalsegOrganTypeV2
        # self.totalseg_mapping = self.totalseg_decoder.load(merge_level)
        
    def totalseg2mask(self, label, organtype: List[OrganTypeBase]=None):
        if organtype is None: return label
        label_ = np.zeros_like(label) if isinstance(label, np.ndarray) else torch.zeros_like(label) 
        for organ in organtype:
            label_index = organ.label
            totalseg_indices = self.totalseg_decoder.organtypes[organ.name]
            if isinstance(totalseg_indices, int): totalseg_indices = [totalseg_indices]
            for totalseg_index in totalseg_indices:
                label_[label == totalseg_index] = label_index
        # import SimpleITK as sitk
        # sitk.WriteImage(sitk.GetImageFromArray(label[0].numpy().astype(np.uint8)), "/ailab/user/dailinrui/data/test_before_process.nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(label_[0].numpy().astype(np.uint8)), "/ailab/user/dailinrui/data/test_after_process.nii.gz")
        return label_


def identity(x, *a, **b): return x


def conserve_only_certain_labels(label,
                                 designated_labels=[1, 2, 3, 5, 6, 10, 55, 56, 57, 104],
                                 totalseg_version="v1"):
    if isinstance(label, np.ndarray):
        if designated_labels is None:
            return label.astype(np.uint8)
        label_ = np.zeros_like(label)
    elif isinstance(label, torch.Tensor):
        if designated_labels is None:
            return label.long()
        label_ = torch.zeros_like(label)
    for il, l in enumerate(designated_labels):
        label_[label == l] = il + 1
    return label_


def maybe_mkdir(p, destory_on_exist=False):
    if path.exists(p) and destory_on_exist:
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)
    return pb.Path(p)

            
def get_date(date_string):
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    _date_ymd = re.findall(date_pattern, date_string)[0]
    date = datetime.strptime(_date_ymd, "%Y-%m-%d") if len(_date_ymd) > 1 else None
    return date

def parse(i, target_res, raw=False):
    img = nib.load(i).dataobj[:].transpose(2, 1, 0)
    if raw:
        return img, np.zeros((3,))
    resize_coeff = np.array(target_res) / np.array(img.shape)
    resized = ndimage.zoom(img, resize_coeff, order=3)
    return resized, resize_coeff


def _mp_prepare(process_dict, save_dir, target_res, pid, raw=False):
    cumpred = 0
    dummy_dir = maybe_mkdir(f"data/temp/{pid}", destory_on_exist=True)
    for k, patient_imaging_history in process_dict.items():
        cumpred += 1
        _latent = defaultdict(list)
        # if path.exists(save_dir / f"case_{k}.npz"): continue
        valid_imaging_histories = [_ for _ in sorted(patient_imaging_history, key=lambda x: get_date(x["time"]))
                                   if len(_["abd_imagings"]) > 0]
        for img_index, img in enumerate(valid_imaging_histories):
            if len(img["abd_imagings"]) == 0: continue
            _latent["date"].append(get_date(img["time"]))
            parsed, coeff = parse(img["abd_imagings"][0], target_res, raw)
            _latent["resize_coeff"].append(coeff)
            _latent["img"].append(parsed)
        if len(_latent["date"]) == 0: continue
        
        dates = np.stack(_latent["date"], axis=0)
        if not raw:
            imgs = np.stack(_latent["img"], axis=0)
            coeffs = np.stack(_latent["resize_coeff"], axis=0)
            np.savez(dummy_dir / f"case_{k}.npz", date=dates, img=imgs, resize_coeff=coeffs)
            print(f"<{pid}> is processing {k}: {cumpred}/{len(process_dict)} cases {coeffs[0].tolist()}", end="\r")
        else:
            np.savez(dummy_dir / f"case_{k}.npz", *_latent["img"], date=dates)
            print(f"<{pid}> is processing {k}: {cumpred}/{len(process_dict)} cases {_latent['img'][0].shape}", end="\r") 
        shutil.copyfile(dummy_dir / f"case_{k}.npz", save_dir / f"case_{k}.npz")
        os.remove(dummy_dir / f"case_{k}.npz")
    shutil.rmtree(dummy_dir)
    
    
def check_validity(file_ls):
    broken_ls = []
    for ifile, file in enumerate(file_ls):
        try:
            np.load(file)
        except Exception as e:
            print(f"{file} raised exception {e}, reprocessing")
            broken_ls.append(file.name.split("_")[1].split(".")[0])
        print(f"<{os.getpid()}> is processing {ifile}/{len(file_ls)}", end="\r")
    return broken_ls


def window_norm(image, window_pos=60, window_width=360):
    window_min = window_pos - window_width // 2
    image = (image - window_min) / window_width
    image[image < 0] = 0
    image[image > 1] = 1
    return image


def load_or_write_split(basefolder, force=False, **splits):
    splits_file = os.path.join(basefolder, "splits.json")
    if os.path.exists(splits_file) and not force:
        with open(splits_file, "r") as f:
            splits = json.load(f)
    else:
        with open(splits_file, "w") as f:
            json.dump(splits, f, indent=4)
    splits = list(splits.get(_) for _ in ["train", "val", "test"])
    return splits


class TorchioSequentialTransformer:
    def __init__(self, d: MutableMapping, force_include=False):
        self.transform_keys = d.keys()
        self.transforms = d.values()
        self.force_include = force_include
        
    def __call__(self, x: tio.Subject):
        for k, tr in zip(self.transform_keys, self.transforms):
            x = tr(x) if not self.force_include else tr(x, include=[k])
        return x


class TorchioBaseResizer(tio.transforms.Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def _interpolate(x, scale_coef, mode="trilinear"):
        x_rsz = torch.nn.functional.interpolate(x[None].float(), scale_factor=scale_coef, mode=mode)[0]
        if mode == "nearest":
            x_rsz = x_rsz.round()
        return x_rsz
        
    def apply_transform(self, data: tio.Subject):
        # data: c h w d
        subject_ = {k: v.data for k, v in data.items()}
        type_ = {k: v.type for k, v in data.items()}
        class_ = {k: tio.ScalarImage if isinstance(v, tio.ScalarImage) else tio.LabelMap for k, v in data.items()}
        
        local_spacing = np.array(data[list(subject_.keys())[0]]["spacing"])
        scale_coef = tuple(local_spacing / local_spacing.mean())[::-1]
        
        subject_ = {k: class_[k](tensor=self._interpolate(v, scale_coef, mode="nearest" if type_[k] == "label" else "trilinear"), type=type_[k]) for k, v in subject_.items()}
        return tio.Subject(subject_)


class TorchioForegroundCropper(tio.transforms.Transform):
    def __init__(self, crop_level="all", crop_kwargs=None, crop_anchor=None,
                 *args, **kwargs):
        self.crop_level = crop_level
        self.crop_kwargs = crop_kwargs
        self.crop_anchor = crop_anchor
        super().__init__(*args, **kwargs)

    def apply_transform(self, data: tio.Subject):
        # data: c h w d
        subject_ = {k: v.data for k, v in data.items()}
        type_ = {k: v.type for k, v in data.items()}
        class_ = {k: tio.ScalarImage if isinstance(v, tio.ScalarImage) else tio.LabelMap for k, v in data.items()}

        if self.crop_level == "all":
            return data

        if self.crop_level == "patch":
            image_ = subject_[self.crop_anchor]
            output_size = self.crop_kwargs["output_size"]
            
            pw = max((output_size[0] - image_.shape[1]) // 2 + 3, 0)
            ph = max((output_size[1] - image_.shape[2]) // 2 + 3, 0)
            pd = max((output_size[2] - image_.shape[3]) // 2 + 3, 0)
            image_ = torch.nn.functional.pad(image_, (pd, pd, ph, ph, pw, pw), mode='constant', value=0)

            (c, w, h, d) = image_.shape
            w1 = np.random.randint(0, w - output_size[0])
            h1 = np.random.randint(0, h - output_size[1])
            d1 = np.random.randint(0, d - output_size[2])
            
            padder = identity if pw + ph + pd == 0 else lambda x: torch.nn.functional.pad(x, (pd, pd, ph, ph, pw, pw), mode='constant', value=0)
            cropper = [slice(w1, w1 + output_size[0]), slice(h1, h1 + output_size[1]), slice(d1, d1 + output_size[2])]
            subject_ = {k: class_[k](tensor=padder(v)[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
            
        outline = self.crop_kwargs.get("outline", [0] * 6)
        if isinstance(outline, int): outline = [outline] * 6
        if len(outline) == 3: outline = reduce(lambda x, y: x + y, zip(outline, outline))
        if self.crop_level == "image_foreground":
            image_ = subject_[self.crop_anchor]
            s1, e1 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(-1).any(-1).any(0))[0][[0, -1]]
            s2, e2 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(1).any(-1).any(0))[0][[0, -1]]
            s3, e3 = torch.where((image_ >= self.crop_kwargs.get('foreground_hu_lb', 0)).any(1).any(1).any(0))[0][[0, -1]]
            cropper = [slice(max(0, s1 - outline[0]), min(e1 + 1 + outline[1], image_.shape[1])),
                       slice(max(0, s2 - outline[2]), min(e2 + 1 + outline[3], image_.shape[2])),
                       slice(max(0, s3 - outline[4]), min(e3 + 1 + outline[5], image_.shape[3]))]
            subject_ = {k: class_[k](tensor=v[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
        
        if self.crop_level == "mask_foreground":
            mask_ = conserve_only_certain_labels(subject_[self.crop_anchor], self.crop_kwargs.get("foreground_mask_label", None))
            s1, e1 = torch.where(mask_.any(-1).any(-1).any(0))[0][[0, -1]]
            s2, e2 = torch.where(mask_.any(1).any(-1).any(0))[0][[0, -1]]
            s3, e3 = torch.where(mask_.any(1).any(1).any(0))[0][[0, -1]]
            cropper = [slice(max(0, s1 - outline[0]), min(e1 + 1 + outline[1], mask_.shape[1])),
                       slice(max(0, s2 - outline[2]), min(e2 + 1 + outline[3], mask_.shape[2])),
                       slice(max(0, s3 - outline[4]), min(e3 + 1 + outline[5], mask_.shape[3]))]
            subject_ = {k: class_[k](tensor=v[:, cropper[0], cropper[1], cropper[2]], type=type_[k]) for k, v in subject_.items()}
            
        return tio.Subject(subject_)
            
            


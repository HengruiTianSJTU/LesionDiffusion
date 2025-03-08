from abc import abstractmethod
from torch.utils.data import Dataset, IterableDataset

import os
import h5py
import json
import torch
import random
import SimpleITK as sitk
import torchio as tio
from ldm.data.utils import TorchioForegroundCropper, identity, TorchioBaseResizer, TorchioSequentialTransformer


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass
    

class MSDDataset(Dataset):
    def __init__(self, base_folder, mapping={}, split="train", max_size=None, resize_to=(96,)*3, force_rewrite_split=False, info={}):
        self.load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        self.get_spacing = lambda x: sitk.ReadImage(x).GetSpacing()
        self.transforms = dict(
            resize_base=TorchioBaseResizer(),
            resize=tio.Resize(resize_to) if resize_to is not None else tio.Lambda(identity),
            crop=TorchioForegroundCropper(crop_level="mask_foreground", 
                                          crop_anchor="totalseg",
                                          crop_kwargs=dict(foreground_hu_lb=1e-3,
                                                            foreground_mask_label=None,
                                                            outline=(0, 0, 0))),
            normalize_image=tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=None, include=["image"]),
            normalize_mask=tio.RemapLabels(mapping, include=["mask"])
        )

        self.split = split
        self.base_folder = base_folder
        if not os.path.exists(f"{base_folder}/train.list") or force_rewrite_split:
            train_val_cases = os.listdir(f"{base_folder}/imagesTr")
            random.shuffle(train_val_cases)
            train_cases = train_val_cases[:round(0.8 * len(train_val_cases))]
            val_cases = train_val_cases[round(0.8 * len(train_val_cases)):]
            test_cases = os.listdir(f"{base_folder}/imagesTs")
            
            for spt in ["train", "val", "test"]:
                with open(f"{base_folder}/{spt}.list", "w") as fp:
                    for c in locals().get(f"{spt}_cases"):
                        fp.write(c + "\n")
        
        for spt in ["train", "val", "test"]:
            with open(f"{base_folder}/{spt}.list") as fp:
                self.__dict__[f"{spt}_keys"] = [_.strip() for _ in fp.readlines()]
        else:
            self.split_keys = getattr(self, f"{split}_keys")[:max_size]
            
        self.__dict__.update(info)
            
    def __getitem__(self, idx):
        item = self.split_keys[idx] if isinstance(idx, int) else idx
        
        if self.split in ["train", "val"]:
            image, mask, totalseg = map(lambda x: self.load_fn(os.path.join(self.base_folder, x, item)), ["imagesTr", "labelsTr", "totalsegTr"])
            spacing = self.get_spacing(os.path.join(self.base_folder, "labelsTr", item))
            subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing), 
                                  mask=tio.LabelMap(tensor=mask[None], spacing=spacing),
                                  totalseg=tio.LabelMap(tensor=totalseg[None], spacing=spacing))
        if self.split == "test":
            image = self.load_fn(os.path.join(self.base_folder, "imagesTs", item))
            spacing = self.get_spacing(os.path.join(self.base_folder, "imagesTs", item))
            subject = tio.Subject(image=tio.ScalarImage(tensor=image[None], spacing=spacing))
        # resize based on spacing
        subject = self.transforms["resize_base"](subject)
        # crop
        subject = self.transforms["crop"](subject)
        ori_size = subject.image.data.shape
        # normalize
        subject = self.transforms["normalize_image"](subject)
        subject = self.transforms["normalize_mask"](subject)
        # resize
        subject = self.transforms["resize"](subject)
        # random aug
        subject = self.transforms.get("augmentation", tio.Lambda(identity))(subject)
        subject = {k: v.data for k, v in subject.items()} | {'casename': self.split_keys[idx] if isinstance(idx, int) else idx, "ori_size": ori_size}

        return subject
    
    
class GeneratedDataset(Dataset):
    def __init__(self, base_folder, max_size=None, split="all", seed=1234, val_num=5):
        self.base = base_folder
        with open(os.path.join(base_folder, "dataset.json")) as f:
            self.desc = json.load(f)
        self.split = split
        self.keys = self.desc["keys"]
        self.data = self.desc["data"]
        self.format = self.desc["format"]
        self.all_keys = list(self.data.keys())
        
        if self.split != "all":
            random.seed(seed)
            random.shuffle(self.all_keys)
            self.train_keys = self.all_keys[:-val_num]
            self.val_keys = self.all_keys[-val_num:]
        
        self.transforms = TorchioSequentialTransformer({})
        self.split_keys = getattr(self, f"{split}_keys")[:max_size]
        
    def __len__(self):
        return len(self.split_keys)
    
    def _read_nifti(self, x):
        return self._possible_expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(x)))
    
    def _read_h5(self, x):
        s = dict()
        h5 = h5py.File(x)
        for k in h5.keys():
            s[k] = self._possible_expand_dims(h5[k][:])
        return s
    
    @staticmethod
    def _possible_expand_dims(x):
        if hasattr(x, "ndim") and x.ndim == 3: x = x[None]
        return torch.tensor(x)
    
    def __getitem__(self, idx):
        casename = self.split_keys[idx]
        item = self.data[casename]
        sample = {}
        image_modalities = []
        for k in self.keys:
            if k not in self.format:
                sample[k] = item[k]
                continue
            if self.format[k] == "raw": sample[k] = item[k]
            elif self.format[k] == '.nii.gz': sample[k] = self._read_nifti(item[k])
            elif self.format[k] == '.h5': sample = sample | self._read_h5(item[k])
            if self.format[k] != 'raw': image_modalities.append(k)
        
        if len(image_modalities) > 0:
            subject = tio.Subject(**{k: tio.ScalarImage(tensor=sample[k]) for k in image_modalities})
            self.transforms(subject)
            sample.update({k: v.data for k, v in subject.items()})
        return sample
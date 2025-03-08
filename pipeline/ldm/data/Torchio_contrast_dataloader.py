import concurrent.futures
import copy
import json
import os
import re
import random
import sys
from typing import Any
import collections
import traceback
import shutil
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytorch_lightning as pl
from multiprocessing import Pool, cpu_count
import scipy.ndimage as ndimage

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler, _utils

from data_process_func import *

terminal=sys.stdout
totalseg_class = {
    "1": "spleen",
    "2": "kidney right",
    "3": "kidney left",
    "4": "gallbladder",
    "5": "liver",
    "6": "stomach",
    "7": "aorta",
    "8": "inferior vena cava",
    "9": "portal vein and splenic vein",
    "10": "pancreas",
    "11": "adrenal gland right",
    "12": "adrenal gland left",
    "13": "lung upper lobe left",
    "14": "lung lower lobe left",
    "15": "lung upper lobe right",
    "16": "lung middle lobe right",
    "17": "lung lower lobe right",
    "18": "vertebrae L5",
    "19": "vertebrae L4",
    "20": "vertebrae L3",
    "21": "vertebrae L2",
    "22": "vertebrae L1",
    "23": "vertebrae T12",
    "24": "vertebrae T11",
    "25": "vertebrae T10",
    "26": "vertebrae T9",
    "27": "vertebrae T8",
    "28": "vertebrae T7",
    "29": "vertebrae T6",
    "30": "vertebrae T5",
    "31": "vertebrae T4",
    "32": "vertebrae T3",
    "33": "vertebrae T2",
    "34": "vertebrae T1",
    "35": "vertebrae C7",
    "36": "vertebrae C6",
    "37": "vertebrae C5",
    "38": "vertebrae C4",
    "39": "vertebrae C3",
    "40": "vertebrae C2",
    "41": "vertebrae C1",
    "42": "esophagus",
    "43": "trachea",
    "44": "heart myocardium",
    "45": "heart atrium left",
    "46": "heart ventricle left",
    "47": "heart atrium right",
    "48": "heart ventricle right",
    "49": "pulmonary artery",
    "50": "brain",
    "51": "iliac artery left",
    "52": "iliac artery right",
    "53": "iliac vena left",
    "54": "iliac vena right",
    "55": "small bowel",
    "56": "duodenum",
    "57": "colon",
    "58": "rib left 1",
    "59": "rib left 2",
    "60": "rib left 3",
    "61": "rib left 4",
    "62": "rib left 5",
    "63": "rib left 6",
    "64": "rib left 7",
    "65": "rib left 8",
    "66": "rib left 9",
    "67": "rib left 10",
    "68": "rib left 11",
    "69": "rib left 12",
    "70": "rib right 1",
    "71": "rib right 2",
    "72": "rib right 3",
    "73": "rib right 4",
    "74": "rib right 5",
    "75": "rib right 6",
    "76": "rib right 7",
    "77": "rib right 8",
    "78": "rib right 9",
    "79": "rib right 10",
    "80": "rib right 11",
    "81": "rib right 12",
    "82": "humerus left",
    "83": "humerus right",
    "84": "scapula left",
    "85": "scapula right",
    "86": "clavicula left",
    "87": "clavicula right",
    "88": "femur left",
    "89": "femur right",
    "90": "hip left",
    "91": "hip right",
    "92": "sacrum",
    "93": "face",
    "94": "gluteus maximus left",
    "95": "gluteus maximus right",
    "96": "gluteus medius left",
    "97": "gluteus medius right",
    "98": "gluteus minimus left",
    "99": "gluteus minimus right",
    "100": "autochthon left",
    "101": "autochthon right",
    "102": "iliopsoas left",
    "103": "iliopsoas right",
    "104": "urinary bladder"
}

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 iter_num=2000, num_workers=None, class_num=1024,
                 crop_shape=[]):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers
        self.class_num = class_num
        self.iter_num = iter_num
        self.crop_shape = crop_shape
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["train"] = test
            self.test_dataloader = self._test_dataloader
        self.collate_fn = _utils.collate.default_collate

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.trainData = LesionDataset(
                                iter_num=self.iter_num,
                                class_num = self.class_num,
                                batch_size= self.batch_size,
                                random_sample=True,
                                **self.dataset_configs["train"],
                                transform=transforms.Compose([
                                    RandomCrop(output_size=self.crop_shape, fg_focus_prob=1, class_determin=[2], target_mode='label'),
                                    MultiThreshNormalized(include=['image'], thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.8,1]]),
                                    # RandomFlip(include=['image', 'label'], prob=0.3, axes=[2]),
                                    # RandomElasticDeformation(num_control_points=[5,5,5], 
                                    #     max_displacement=[5,5,5], include=['image','label'], prob=0),
                                    # RandomAnisotropy(axes=[0], downsampling=(1, 3), include=['image'], prob=0.0),
                                    # RandomAffine(scales=[0.2,0.2,0.2], degrees=[10,10,10],
                                    #     include=['image','label'], prob=0.2),
                                    ToTensor(concat_key=[]),
                                ]),
                                load_numworker = self.num_workers,
                                normalize=False
                                )
        self.validData = LesionDataset(
                                iter_num = len(self.dataset_configs["validation"]['image_list']),
                                class_num = self.class_num,
                                batch_size= self.batch_size,
                                random_sample=True,
                                **self.dataset_configs["validation"],
                                transform=transforms.Compose([
                                    RandomCrop(output_size=self.crop_shape, fg_focus_prob=0.8, class_determin=[2], target_mode='label'),
                                    MultiThreshNormalized(include=['image'], thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.8,1]]),
                                    # PadToScale(scale=32, mode='image'),
                                    ToTensor(),
                                ]),
                                load_numworker = 8,
                                normalize=False,
                                )
                
    def _train_dataloader(self):                
        return DataLoader(self.trainData, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def _val_dataloader(self, shuffle=False, batch_size=1):
        return DataLoader(self.validData, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=self.collate_fn)
    
    def _test_dataloader(self, shuffle=False, batch_size=1):                
        return DataLoader(self.trainData, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
def create_sample(image_path=False, label_path=False, seg_path=False, coarseg_path=False, \
    distance_path=False, type_path=False, class_num=2, normalize=False, \
    thresh_ls=[], norm_ls=[], respacing=False, target_spacing=1):#, transpose=True):
    sample = {}
    if image_path:
        image = torch.from_numpy(load_volfile(image_path, respacing=respacing, target_spacing=target_spacing, mode='image')).to(torch.float16)
        if len(image.shape)==3:
            image = image.unsqueeze(0)
        elif len(image.shape) != 4:
            raise ValueError('please input the right image dimension!! e.g. 3, 4')
        if normalize:
            for i in range(len(thresh_ls)):
                if i ==0:
                    tensor=img_multi_thresh_normalized_torch(image, thresh_ls[i], norm_ls[i], data_type=torch.float16)
                else:
                    tensor=torch.cat((tensor, img_multi_thresh_normalized_torch(image, thresh_ls[i], norm_ls[i], data_type=torch.float16)),0)
            image = tensor
        sample['image']=image
        sample['image_path']=image_path
        
    if label_path:
        label = torch.from_numpy(load_volfile(label_path, respacing=respacing, target_spacing=target_spacing, mode='label').astype(np.uint8))
        if label.shape[-3:] != sample['image'].shape[-3:]:
            if np.max(np.abs(np.array(label.shape[-3:])-np.array(sample['image'].shape[-3:])))<=3:
                label = F.interpolate(label.unsqueeze(0).unsqueeze(0), size=sample['image'].shape[-3:], mode='nearest').squeeze()
            else:
                raise ValueError(label.shape, label_path, sample['image'].shape, image_path)
        # label = F.one_hot(label.long(), class_num).permute(3, 0, 1, 2).to(torch.uint8)
        if len(label.shape)==3:
            label = label.unsqueeze(0)
        elif len(label.shape) != 4:
            raise ValueError('please input the right label dimension!! e.g. 3, 4')
        sample['label']=label
        sample['label_path']=label_path
    if seg_path:
        seg = torch.from_numpy(load_volfile(seg_path, respacing=respacing, target_spacing=target_spacing, mode='label').astype(np.uint8))
        # coarseg = F.one_hot(torch.from_numpy(coarseg).type(torch.LongTensor), class_num).permute(3, 0, 1, 2).to(torch.uint8)
        if len(seg.shape)==3:
            seg = seg.unsqueeze(0)
        elif len(seg.shape) != 4:
            raise ValueError('please input the right image dimension!! e.g. 3, 4')
        sample['seg']=seg
        sample['seg_path']=seg_path
    if coarseg_path:
        coarseg = torch.from_numpy(load_volfile(coarseg_path, respacing=respacing, target_spacing=target_spacing, mode='label').astype(np.uint8))
        # coarseg = F.one_hot(torch.from_numpy(coarseg).type(torch.LongTensor), class_num).permute(3, 0, 1, 2).to(torch.uint8)
        if len(coarseg.shape)==3:
            coarseg = coarseg.unsqueeze(0)
        elif len(coarseg.shape) != 4:
            raise ValueError('please input the right image dimension!! e.g. 3, 4')
        sample['coarseg']=coarseg
        sample['coarseg_path']=coarseg_path
    if distance_path:
        distance = torch.from_numpy(load_volfile(distance_path, respacing=respacing, target_spacing=target_spacing, mode='label').astype(np.uint8))
        if len(distance.shape)==3:
            distance = distance.unsqueeze(0)
        sample['distance']=distance
        sample['distance_path']=distance_path
    if type_path:
        with open(type_path, 'r') as file:
            type_data = json.load(file)
        sample['type'] = type_data
        sample['type_path'] = type_path
    return sample

def process_image(i, cur_image_list, label_list, seg_list, coarseg_list, distance_list, type_list,
                  class_num, normalize, thresh_ls, norm_ls, respacing=False, target_spacing=1):
    image_path = cur_image_list[i]
    if label_list:
        label_path = label_list[i]
    else:
        label_path = False
    if seg_list:
        seg_path = seg_list[i]
    else: 
        seg_path = False
    if coarseg_list:
        coarseg_path = coarseg_list[i]
    else: 
        coarseg_path = False
    if distance_list:
        distance_path = distance_list[i]
    else:
        distance_path = False
    if type_list:
        type_path = type_list[i]
    else:
        type_path = False
    return image_path, create_sample(image_path, label_path,seg_path, coarseg_path, distance_path, type_path, class_num, 
                     normalize, thresh_ls, norm_ls, respacing=respacing, target_spacing=target_spacing)


class LesionDataset(Dataset):
    def __init__(self, iter_num=0, batch_size=0, class_num=3, num=None, transform=None, random_sample=False, 
                 load_memory=False, memory_length=0, image_list=[], label_list=False, type_list=False, seg_list=False,
                 coarseg_list=False, distance_list=False, volume_statics_dic=False, load_numworker=32, max_mask_num=None,
                 normalize=False, thresh_ls=[], norm_ls=[], respacing=False, crop_shape=None, target_spacing=1):
        """
        Args:
            iter_num (int): Number of iterations for the dataloader.
            batch_size (int): Size of each batch.
            class_num (int): Number of classes.
            num (int, optional): Number of samples to load. If None, load all samples.
            transform (callable, optional): Optional transform to be applied on a sample.
            random_sample (bool): Whether to sample randomly.
            load_memory (bool): Whether to load data into memory. If False, data will be loaded on-the-fly.
            memory_length (int): Number of samples to keep in memory.
            image_list (list): List of image file paths.
            label_list (list or bool): List of label file paths or False if not provided.
            coarseg_list (list or bool): List of coarseg file paths or False if not provided.
            distance_list (list or bool): List of distance file paths or False if not provided.
            volume_statics_dic (str or bool): Path to volume statistics dictionary or False if not provided.
            load_numworker (int): Number of workers to use for loading data.
            normalize (bool): Whether to normalize the images.
            thresh_ls (list): List of threshold values for normalization. Could be empty list.
            norm_ls (list): List of normalization values. Could be empty list.
            respacing (bool): Whether to respace the images. If False, images will not be respaced.
            target_spacing (int): Target spacing value for resampling.
        """
        # Initialize the data loader with various parameters and lists
        self.image_task_dic = {}  # Dictionary to store processed image samples
        self.cur_image_list = []  # List to store currently loaded images
        self.left_image_list = []  # List to store remaining images to be loaded

        # Assigning input parameters to class variables
        self._iternum = iter_num
        self.batch_size = batch_size
        if max_mask_num is None:
            if seg_list:
                self.transform = transforms.Compose([
                    CenterCrop(output_size=crop_shape, fg_focus_prob=1, class_determin=[2], target_mode='label'),
                    SegToTensor()
                ])
            else:
                self.transform = transforms.Compose([
                    CenterCrop(output_size=crop_shape, fg_focus_prob=1, class_determin=[2], target_mode='label'),
                    MultiThreshNormalized(include=['image'], thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.8,1]]),
                    ToTensor(concat_key=[])
                ])
        else: 
            if seg_list: self.transform = transforms.Compose([TotalToTensor(max_mask_num, crop_shape, concat_key=[])])
            else:
                self.transform = transforms.Compose([
                    MultiThreshNormalized(include=['image'], thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.8,1]]),
                    TotalToTensor(max_mask_num, crop_shape, concat_key=[])])

        self.random_sample = random_sample
        self.load_memory = load_memory
        self.memory_length = memory_length
        self.class_num = class_num
        self.full_image_list = read_file_list(image_list)  # Reading the full image list from the given file list
        self.load_numworker = load_numworker
        self.normalize = normalize
        self.thresh_ls = thresh_ls
        self.norm_ls = norm_ls
        self.respacing = respacing
        self.target_spacing = target_spacing
        
        # Reading the label list if provided
        if label_list:
            self.label_list = read_file_list(label_list)
        else:
            self.label_list = False
            
        # Reading the seg list if provided
        if seg_list:
            self.seg_list = read_file_list(seg_list)
        else:
            self.seg_list = False

        # Reading the coarseg list if provided
        if coarseg_list:
            self.coarseg_list = read_file_list(coarseg_list)
        else:
            self.coarseg_list = False

        # Reading the distance list if provided
        if distance_list:
            self.distance_list = read_file_list(distance_list)
        else:
            self.distance_list = False

        # Loading volume statistics dictionary if provided
        if volume_statics_dic:
            with open(volume_statics_dic, 'r') as f:
                self.volume_statics_dic = json.load(f)
        else:
            self.volume_statics_dic = False
            
        # Reading the type list if provided
        if type_list:
            self.type_list = read_file_list(type_list)
        else:
            self.type_list = False
        
        # Loading images into memory if load_memory is set to True
        if self.load_memory:
            if self.memory_length:
                cur_sample_index = random.sample(list(range(len(self.full_image_list))), self.memory_length)
            else:
                cur_sample_index = list(range(len(self.full_image_list)))
            for i in cur_sample_index:
                self.cur_image_list.append(self.full_image_list[i])
            if label_list:
                self.cur_label_list = [self.label_list[i] for i in cur_sample_index]
            else:
                self.cur_label_list = False
            if seg_list:
                self.cur_seg_list = [self.seg_list[i] for i in cur_sample_index]
            else:
                self.cur_seg_list = False
            if coarseg_list:
                self.cur_coarseg_list = [self.coarseg_list[i] for i in cur_sample_index]
            else:
                self.cur_coarseg_list = False
            if distance_list:
                self.cur_distance_list = [self.distance_list[i] for i in cur_sample_index]
            else:
                self.cur_distance_list = False
            if type_list:
                self.cur_type_list = [self.type_list[i] for i in cur_sample_index]
            else:
                self.cur_type_list = False
        
            # Using ThreadPoolExecutor to process images concurrently
            with concurrent.futures.ThreadPoolExecutor(self.load_numworker) as executor:
                futures = {executor.submit(process_image, i, self.cur_image_list, self.cur_label_list, self.cur_seg_list, self.cur_coarseg_list, 
                            self.cur_distance_list, self.cur_type_list, self.class_num, self.normalize, self.thresh_ls, self.norm_ls, 
                            self.respacing, self.target_spacing): i for i in range(len(self.cur_image_list))}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
                    i = futures[future]
                    try:
                        image_path, image_sample = future.result()
                        self.image_task_dic[image_path] = image_sample
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')
            
            self.cur_image_list = [key for key in self.image_task_dic.keys()]
            self.left_image_list = list(set(self.full_image_list) - set(self.cur_image_list))
            random.shuffle(self.left_image_list)
        
        if num is not None:
            self.full_image_list = self.full_image_list[:num]
                
        print("total {} samples".format(len(self.full_image_list)))

    def __len__(self):
        # Returns the length of the dataset
        if self._iternum:
            return self._iternum
        else:
            if self.load_memory:
                return len(self.cur_image_list)
            else:
                return len(self.full_image_list)

    def updata_memory(self):
        # Updates the in-memory samples by replacing half of them with new ones
        new_samples = {}
        new_image_list = []
        
        for _ in range(len(self.cur_image_list) // 2):
            out_image_name = self.cur_image_list.pop(0)
            del self.image_task_dic[out_image_name]
            self.left_image_list.append(out_image_name)

            in_image_name = self.left_image_list.pop(0)
            new_image_list.append(in_image_name)
            self.cur_image_list.append(in_image_name)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for in_image_name in new_image_list:
                in_idx = self.full_image_list.index(in_image_name)
                label_path = self.label_list[in_idx] if self.label_list else False
                seg_path = self.seg_list[in_idx] if self.seg_list else False
                coarseg_path = self.coarseg_list[in_idx] if self.coarseg_list else False
                distance_path = self.distance_list[in_idx] if self.distance_list else False
                type_list = self.type_list[in_idx] if self.type_list else False

                futures.append(executor.submit(create_sample, in_image_name, label_path, seg_path, 
                            coarseg_path, distance_path, type_list, self.class_num, self.normalize,
                            self.thresh_ls, self.norm_ls, self.respacing, self.target_spacing))

            pbar = tqdm(total=len(futures), desc="Updating memory", dynamic_ncols=True)

            for future, in_image_name in zip(futures, new_image_list):
                try:
                    nsample = future.result()
                    new_samples[nsample['image_path']] = nsample
                    pbar.update(1)
                except Exception as e:
                    print(f"An error occurred with image '{in_image_name}': {e}")
                    if in_image_name in self.cur_image_list:
                        self.cur_image_list.remove(in_image_name)

            pbar.close()

        self.image_task_dic.update(new_samples)

    def __getitem__(self, idx):
        # Returns a sample from the dataset
        try:
            if self.load_memory:
                if self.random_sample:
                    idx = random.randint(0, len(self.cur_image_list) - 1)
                image_path = self.cur_image_list[idx]
                sample = copy.deepcopy(self.image_task_dic[image_path])
            else:
                if self.random_sample:
                    idx = random.randint(0, len(self.full_image_list) - 1)
                image_path = self.full_image_list[idx]
                if self.label_list:
                    label_path = self.label_list[idx]
                else:
                    label_path = False
                if self.seg_list:
                    seg_path = self.seg_list[idx]
                else: 
                    seg_path = False
                if self.coarseg_list:
                    coarseg_path = self.coarseg_list[idx]
                else: 
                    coarseg_path = False
                if self.distance_list:
                    distance_path = self.distance_list[idx]
                else:
                    distance_path = False
                if self.type_list:
                    type_path = self.type_list[idx]
                else:
                    type_path = False
                sample = create_sample(image_path, label_path, seg_path, coarseg_path, distance_path, type_path, self.class_num, 
                                       self.normalize, self.thresh_ls, self.norm_ls, self.respacing, self.target_spacing)

            if self.volume_statics_dic:
                sample['volume_statics_dic'] = self.volume_statics_dic[image_path]
            if self.transform:
                sample = self.transform(sample)
            return sample
        
        except Exception as e:
            # Handle the exception (e.g., print an error message)
            folder_path = os.path.dirname(image_path)
            traceback.print_exception(e)
            print(f"Error creating sample: {e} in {folder_path}") 
                       
            # Optionally, you can retry with a different index
            if self.load_memory:
                idx = random.randint(0, len(self.cur_image_list) - 1)
            else:
                idx = random.randint(0, len(self.full_image_list) - 1)
            
            # Recursively call __getitem__ with the new index
            return self.__getitem__(idx)
    
    def verify_dataset(self):
        """
        Verifies the dataset by iterating through all items and checking their shapes.
        """
        iterator = tqdm(range(self.__len__()))
        for idx in iterator:
            try:
                item = self.__getitem__(idx)
                iterator.set_postfix(shape=item["image"].shape)
            except Exception as e:
                print(f"Error loading item at index")
                
    def collate(self, batch, unpack=True):
        elem = batch[0]
        elem_type = type(elem)
        
        if isinstance(elem, collections.abc.Mapping):
            if unpack:
                try:
                    return elem_type({key: self.collate([d[key] for d in batch], unpack = False) for key in elem})
                except TypeError:
                    # The mapping type may not support `__init__(iterable)`.
                    return {key: self.collate([d[key] for d in batch], unpack = False) for key in elem}
            else:
                return batch
        elif isinstance(elem, torch.Tensor):
            return torch.stack(batch)
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

            if isinstance(elem, tuple):
                return [self.collate(samples) for samples in transposed]  # Backwards compatibility.
            else:
                try:
                    return elem_type([self.collate(samples) for samples in transposed])
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [self.collate(samples) for samples in transposed]

        raise TypeError(elem_type)


class VolumeStatics(object):  
    def __init__(self, mode='label', image='image', num_workers=16):  
        self.mode = mode  
        self.image = image  
        self.num_workers = num_workers
        
    def compute_stats(self, class_id, images, labels):
        mask = (labels == class_id)  
        mean = np.mean(images[mask])  
        # std = np.std(images[mask])  
        return class_id, mean.item()
    
    def __call__(self, sample):
        # time0 = time.time() 
        # 提取标签和图像数据  
        labels = sample[self.mode].numpy()  
        images = sample[self.image].numpy()  
        # 将标签转换为uint8，并排除背景（假设背景为0）  
        unique_class_ids = np.unique(labels.astype(np.uint8))  
        unique_class_ids = unique_class_ids[unique_class_ids != 0]  
          
        # 使用Pool并行计算每个类别的统计信息
        volume_statics_dic = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for class_id in unique_class_ids:
                # Submit the create_sample task to the executor
                futures.append(executor.submit(self.compute_stats, class_id, images, labels))

            # Wait for all futures to complete
            for future in futures:
                class_id, mean = future.result()
                volume_statics_dic[str(class_id.item())] = [mean] 
        # 将统计信息添加到样本中  
        sample['volume_statics_dic'] = volume_statics_dic  
        # 返回处理后的样本 
        # time1 = time.time()
        # print('static time cost:', time1-time0)
        return sample  


class CropBound(object):
    def __init__(self, pad=[0,0,0], mode='label', class_determine=False):
        self.pad = pad
        self.mode = mode
        self.class_determine=class_determine
    def __call__(self, sample):
        if self.mode=='label':
            if self.class_determine:
                file = sample[self.mode].index_select(0, torch.tensor(self.class_determine))
            else:
                file = sample[self.mode][1::]
        else:
            if self.class_determine:
                file = torch.isin(sample[self.mode], torch.tensor(self.class_determine))
            else:
                file = sample[self.mode]
        file = torch.sum(file, dim=0) 
        file_size = file.shape # DWH
        nonzeropoint = torch.as_tensor(torch.nonzero(file))
        maxpoint = torch.max(nonzeropoint, 0)[0].tolist()
        minpoint = torch.min(nonzeropoint, 0)[0].tolist()
        for i in range(len(self.pad)):
            maxpoint[i] = min(maxpoint[i] + self.pad[i], file_size[i])
            minpoint[i] = max(minpoint[i] - self.pad[i], 0)
            if 'bbox' in sample.keys():
                sample['bbox'] -= np.array(minpoint)[np.newaxis,:]
                sample['bbox'][sample['bbox'] < 0] = 0

        sample['minpoint']=minpoint
        sample['maxpoint']=maxpoint
        sample['shape'] = file_size

        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key]=sample[key][:, minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        return sample


class PadToScale(object):
    def __init__(self, scale, mode='label'):
        self.scale = scale
        self.mode = mode

    def _compute_pad(self, size):
        # 计算每个维度需要的padding数量，使其可以被scale整除
        return [int((self.scale - s % self.scale) % self.scale) for s in size]

    def __call__(self, sample):
        file = sample[self.mode][1::]
        file_size = file.shape[-3:]  # 假设tensor的shape为[C, D, H, W]
        
        pad_values = self._compute_pad(file_size)
        # 只在每个维度的一侧进行padding
        pad = (0, pad_values[2], 0, pad_values[1], 0, pad_values[0])
        sample['pad'] = pad
        # 更新sample中的所有tensor
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key] = F.pad(sample[key], pad=pad, mode='constant', value=0)

        return sample


class ExtractCertainClass(object):
    def __init__(self, class_wanted=[1]):
        self.class_wanted = class_wanted
    def __call__(self, sample):
        label = sample['label']
        nlabel = label.index_select(0, torch.tensor([0]+self.class_wanted))
        sample ['label'] = nlabel
        if 'coarseg' in sample:
            ncoarseg = sample['coarseg'].index_select(0, torch.tensor([0]+self.class_wanted))
            sample ['coarseg'] = ncoarseg
                
        return sample

class ExtractCertainClassScribble(object):
    def __init__(self, class_wanted=[1]):
        self.class_wanted = class_wanted
    def __call__(self, sample):
        label = sample['label']
        nlabel = label.index_select(0, torch.tensor([0]+self.class_wanted[1::]))
        sample ['label'] = nlabel
        if 'coarseg' in sample:
            ncoarseg = sample['coarseg'].index_select(0, torch.tensor([0]+self.class_wanted[1::]))
            sample ['coarseg'] = ncoarseg
        return sample

class LabelExist(object):
    def __init__(self, include=[], class_num=0):
        self.class_num = class_num
    def __call__(self, sample):
        label = sample['label']
        nlabel = label.index_select(0, torch.tensor([0]+self.class_wanted[1::]))
        sample ['label'] = nlabel
        if 'coarseg' in sample:
            ncoarseg = sample['coarseg'].index_select(0, torch.tensor([0]+self.class_wanted[1::]))
            sample ['coarseg'] = ncoarseg
        return sample

class RandomNoiseAroundClass(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma
    def __call__(self, sample):
        image = sample['image']
        noise = torch.clamp(self.sigma * torch.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        sample['image'] = image
        return sample


class RandomCrop(object):
    def __init__(self, output_size, fg_focus_prob=0, target_mode='label', class_determin=[], target_type=torch.float32):
        '''
        output_size (int, tuple, list): 裁剪输出的大小。可以是整数（各维度相同），也可以是包含每个维度大小的元组或列表。
        fg_focus_prob (float): 前景聚焦概率。表示是否根据前景类进行裁剪的概率。默认值为0。
        target_mode (str): 指定目标模式的键，通常是样本字典中的一个键。默认值为 'label'。
        class_determin (list): 需要聚焦的前景类列表。用于确定裁剪区域包含指定的类。可以为空
        target_type (torch.dtype): 目标数据类型。用于转换裁剪后的数据类型，默认值为 torch.float32。
        '''
        # assert isinstance(output_size, (int, tuple, list))
        self.output_size = output_size
        self.fg_focus_prob = fg_focus_prob
        self.target_mode = target_mode
        self.class_determin = class_determin
        self.target_type = target_type
        
    def __call__(self, sample):
        # 获取目标模式（如标签）的形状，不包括channel维度（DWH：深度、高度、宽度）
        cshape = sample[self.target_mode].shape[1::]
        if isinstance(cshape, torch.Size):
            cshape = list(cshape)

        # 如果样本的尺寸小于输出尺寸，则对样本进行填充
        if cshape[0] <= self.output_size[0] or cshape[1] <= self.output_size[1] or cshape[2] <= self.output_size[2]:
            # 计算每个维度需要填充的大小
            pw = max(self.output_size[0] - cshape[0]+8, 0)
            ph = max(self.output_size[1] - cshape[1]+8, 0)
            pd = max(self.output_size[2] - cshape[2]+8, 0)
            for key in sample.keys():
                # 对样本的每个张量进行填充
                if torch.is_tensor(sample[key]):
                    sample[key] = F.pad(sample[key], (0, pd, 0, ph, 0, pw), mode='constant', value=0)

        # 获取填充后的样本尺寸（DWH）
        (w, h, d) = sample[self.target_mode].shape[1::]
        min_w, max_w, min_h, max_h, min_d, max_d = 0, w - self.output_size[0], 0, h - self.output_size[1], 0, d - self.output_size[2]

        # 如果指定了前景类，并且样本的目标模式不为空，则计算前景类的边界
        if len(self.class_determin) > 0 and sample[self.target_mode] is not None:
            positions = torch.stack(torch.where(torch.isin(sample[self.target_mode], torch.tensor(self.class_determin))))
            if positions.nelement() > 0:
                # 计算前景类所在区域的最小和最大坐标
                min_w = max(positions[1].min() - self.output_size[0] // 1.5, 0)
                max_w = min(max(0, positions[1].max() - self.output_size[0] // 1.5), w - self.output_size[0])
                min_h = max(positions[2].min() - self.output_size[1] // 1.5, 0)
                max_h = min(max(0, positions[2].max() - self.output_size[1] // 1.5), h - self.output_size[1])
                min_d = max(positions[3].min() - self.output_size[2] // 1.5, 0)
                max_d = min(max(0, positions[3].max() - self.output_size[2] // 1.5), d - self.output_size[2])
        
        # 确保最小和最大坐标在有效范围内
        min_w = int(min(min_w, max_w))
        max_w = int(max(max_w, min_w)) + 1
        min_h = int(min(min_h, max_h))
        max_h = int(max(max_h, min_h)) + 1
        min_d = int(min(min_d, max_d))
        max_d = int(max(max_d, min_d)) + 1
        
        # 随机决定是否进行前景聚焦裁剪
        crop_fg = torch.rand(1) < self.fg_focus_prob
        keep_crop = True
        crop_time = 0

        while keep_crop:
            # 随机选择裁剪区域的起始位置
            w1 = torch.randint(min_w, max_w, (1,)).item()
            h1 = torch.randint(min_h, max_h, (1,)).item()
            d1 = torch.randint(min_d, max_d, (1,)).item()

            # 如果进行前景聚焦裁剪且尝试次数小于20次，则检查裁剪区域
            if crop_fg and crop_time < 20 and sample[self.target_mode] is not None:
                crop_time += 1
                # 获取裁剪区域的标签
                crop_label = sample[self.target_mode][:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
                if self.class_determin:
                    # 检查裁剪区域是否包含所有指定的前景类
                    if all(crop_label.eq(class_id).any() for class_id in self.class_determin):
                        keep_crop = False
                else:
                    # 如果裁剪区域包含任何前景类，则停止裁剪
                    if torch.sum(crop_label) > 0:
                        keep_crop = False
            else:
                keep_crop = False
                
            # 计算裁剪块的中心坐标
            center_w = w1 + self.output_size[0] // 2
            center_h = h1 + self.output_size[1] // 2
            center_d = d1 + self.output_size[2] // 2
            
        sample['crop'] = {'crop_center':(center_w, center_h, center_d), 'full_size': (w, h ,d), 'crop_size': tuple(self.output_size)}

        # 对样本的每个张量进行裁剪并转换为目标数据类型
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key] = sample[key][:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]].to(self.target_type)
        return sample

class CenterCrop(RandomCrop):
    def __init__(self, output_size=None, fg_focus_prob=0, target_mode='label', class_determin=[], target_type=torch.float32):
        super().__init__(output_size, fg_focus_prob, target_mode, class_determin, target_type)
    def __call__(self, sample):
        # 获取目标模式（如标签）的形状，不包括channel维度（DWH：深度、高度、宽度）
        cshape = sample[self.target_mode].shape[1::]
        if isinstance(cshape, torch.Size):
            cshape = list(cshape)

        # 如果样本的尺寸小于输出尺寸，则对样本进行填充
        if cshape[0] <= self.output_size[0] or cshape[1] <= self.output_size[1] or cshape[2] <= self.output_size[2]:
            # 计算每个维度需要填充的大小
            pw = max(self.output_size[0] - cshape[0]+8, 0)
            ph = max(self.output_size[1] - cshape[1]+8, 0)
            pd = max(self.output_size[2] - cshape[2]+8, 0)
            for key in sample.keys():
                # 对样本的每个张量进行填充
                if torch.is_tensor(sample[key]):
                    sample[key] = F.pad(sample[key], (0, pd, 0, ph, 0, pw), mode='constant', value=0)

        # 获取填充后的样本尺寸（DWH）
        (w, h, d) = sample[self.target_mode].shape[1::]
        min_w, max_w, min_h, max_h, min_d, max_d = 0, w - self.output_size[0], 0, h - self.output_size[1], 0, d - self.output_size[2]

        # 如果指定了前景类，并且样本的目标模式不为空，则计算前景类的边界
        if len(self.class_determin) > 0 and sample[self.target_mode] is not None:
            positions = torch.isin(sample[self.target_mode], torch.tensor(self.class_determin))
            if positions.nelement() > 0:
                # 计算前景类所在区域的边界
                positions_np = positions.numpy()
                mask = np.zeros_like(positions_np, dtype=bool)
                mask[positions_np] = True
                
                # 标记连通区域
                labeled_array, num_features = ndimage.label(mask)
                
                # 找到最大连通区域的标签
                if num_features > 1:
                    sizes = ndimage.sum(mask, labeled_array, range(num_features + 1))
                    max_label = np.argmax(sizes[1:]) + 1
                    max_component = (labeled_array == max_label)
                    
                    # 获取最大连通区域的位置
                    max_positions = np.where(max_component)
                    positions = torch.tensor(np.array(max_positions))
                else:
                    only_positions = np.where(mask)
                    positions = torch.tensor(np.array(only_positions))
                                                     
                min_w, max_w = positions[1].min().item(), positions[1].max().item()
                min_h, max_h = positions[2].min().item(), positions[2].max().item()
                min_d, max_d = positions[3].min().item(), positions[3].max().item()

        # 计算 bounding box 的中心
        center_w = (min_w + max_w) // 2
        center_h = (min_h + max_h) // 2
        center_d = (min_d + max_d) // 2

        # 确保裁剪区域不会超出图像边界
        half_output_size = np.array(self.output_size) // 2
        w1 = max(center_w - half_output_size[0], 0)
        h1 = max(center_h - half_output_size[1], 0)
        d1 = max(center_d - half_output_size[2], 0)

        w2 = min(w1 + self.output_size[0], w)
        h2 = min(h1 + self.output_size[1], h)
        d2 = min(d1 + self.output_size[2], d)

        # 如果某个维度小于 output_size，则需要调整起始点
        if (w2 - w1) < self.output_size[0]:
            if center_w - half_output_size[0] < 0:  # 左边超出
                w1 = 0
                w2 = self.output_size[0]
            else:  # 右边超出
                w1 = w - self.output_size[0]
                w2 = w
            center_w = (w1 + w2) // 2  # 更新裁剪中心

        if (h2 - h1) < self.output_size[1]:
            if center_h - half_output_size[1] < 0:  # 上边超出
                h1 = 0
                h2 = self.output_size[1]
            else:  # 下边超出
                h1 = h - self.output_size[1]
                h2 = h
            center_h = (h1 + h2) // 2  # 更新裁剪中心

        if (d2 - d1) < self.output_size[2]:
            if center_d - half_output_size[2] < 0:  # 前边超出
                d1 = 0
                d2 = self.output_size[2]
            else:  # 后边超出
                d1 = d - self.output_size[2]
                d2 = d
            center_d = (d1 + d2) // 2  # 更新裁剪中心

        # 更新 crop 信息
        sample['crop'] = {
            'crop_center': (center_w, center_h, center_d),
            'full_size': (w, h, d),
            'crop_size': (w2 - w1, h2 - h1, d2 - d1),
            'crop_start': (w1, h1, d1)
        }      
          
        # 对样本的每个张量进行裁剪并转换为目标数据类型
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key] = sample[key][:, w1:w2, h1:h2, d1:d2].to(self.target_type)

        return sample
            
class RandomCropDepth(object):
    def __init__(self, output_depth, fg_focus_prob=0, target_mode='label', class_determin=[], target_type=torch.float32):
        assert isinstance(output_depth, (int, tuple, list))
        self.output_depth = output_depth
        self.fg_focus_prob = fg_focus_prob
        self.target_mode = target_mode
        self.class_determin = class_determin
        self.target_type = target_type

    def __call__(self, sample):
        cshape = sample[self.target_mode].shape[1::]  # DWH

        # Pad the sample if necessary
        if cshape[0] <= self.output_depth:
            pd = max(self.output_depth - cshape[0]+2, 0)
            for key in sample.keys():
                if torch.is_tensor(sample[key]):
                    # Only pad along the depth dimension
                    sample[key] = F.pad(sample[key], (0, 0, 0, 0, 0, pd), mode='constant', value=0)

        (d, w, h) = sample[self.target_mode].shape[1::]
        # Keep width (w) and height (h) dimensions unchanged
        min_d, max_d = 0, d - self.output_depth

        # Calculate the bounds for specified classes
        if len(self.class_determin) > 0 and sample[self.target_mode] is not None:
            positions = torch.stack(torch.where(torch.isin(sample[self.target_mode], torch.tensor(self.class_determin))))
            if positions.nelement() > 0:
                min_d = max(positions[1].min() - self.output_depth//2, torch.tensor(0))
                max_d = min(positions[1].max() - self.output_depth//2, torch.tensor(d - self.output_depth))
                if min_d > max_d:
                    min_d = max(max_d - self.output_depth//2, 0)

        crop_fg = torch.rand(1) < self.fg_focus_prob
        keep_crop = True
        crop_time = 0

        while keep_crop:
            d1 = torch.randint(min_d, max_d, (1,)).item()

            if crop_fg and crop_time < 20 and sample[self.target_mode] is not None:
                crop_time += 1
                crop_label = sample[self.target_mode][:, d1:d1 + self.output_depth]
                if self.class_determin:
                    if all(crop_label.eq(class_id).any() for class_id in self.class_determin):
                        keep_crop = False
                else:
                    if torch.sum(crop_label) > 0:
                        keep_crop = False
            else:
                keep_crop = False

        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                # Only crop along the depth dimension, keep width and height unchanged
                sample[key] = sample[key][:, d1:d1 + self.output_depth].to(self.target_type)
        return sample


class RandomBboxCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, fg_focus_prob=0, recall_thresh=0.6, pad=[0,0,0]):
        self.output_size = np.array(output_size)
        self.fg_focus_prob = fg_focus_prob
        self.recall_thresh = recall_thresh
        self.pad = pad
    
    def recall(self, box1, box2):
        '计算三维recall of box2,box=[h_min,w_min,d_min,h_max,w_max,d_max]'
        box1 = np.asarray(box1).reshape([-1,1])
        box2 = np.asarray(box2).reshape([-1,1])
        in_h = min(box1[3], box2[3]) - max(box1[0], box2[0])
        in_w = min(box1[4], box2[4]) - max(box1[1], box2[1])
        in_d = min(box1[5], box2[5]) - max(box1[2], box2[2])
        inter = 0 if in_h<0 or in_w<0 or in_d<0 else in_h*in_w*in_d
        bbox2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1])*(box2[5] - box2[2])
        recall = inter / bbox2_volume
        return recall
    def __call__(self, sample):
        cshape = sample['image'].shape[1::] # DWH
        if 'label' in sample.keys():
            sample['real_bbox'] = np.array(get_bound_coordinate(sample['label'][1].numpy()))
        # pad the sample if necessary
        if cshape[0] <= self.output_size[0] or cshape[1] <= self.output_size[1] or cshape[2] <= \
                self.output_size[2]:
            #print(cshape)
            orishape = cshape
            pw = max((self.output_size[0] - cshape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - cshape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - cshape[2]) // 2 + 3, 0)
            for key in sample.keys():
                if torch.is_tensor(sample[key]):
                    sample[key] = F.pad(sample[key], (pd, pd, ph, ph, pw, pw,0 ,0), mode='constant', value=0)
                sample['real_bbox'] += np.array([pd, ph, pw])
        (w, h, d) = sample['image'].shape[1::]
        crop_fg = torch.rand(1)<self.fg_focus_prob
        keep_crop = True
        crop_time = 0
        while keep_crop:
            w1 = np.random.randint(- self.pad[0], w- self.output_size[0]+self.pad[0])
            h1 = np.random.randint(- self.pad[1], h- self.output_size[1]+self.pad[1])
            d1 = np.random.randint(- self.pad[2], d- self.output_size[2]+self.pad[2])
            if crop_fg and crop_time<1000:
                crop_time += 1
                crop_bbox = [w1,h1,d1,w1 + self.output_size[0], h1 + self.output_size[1], d1 + self.output_size[2]]
                if self.recall(crop_bbox, sample['real_bbox'])>self.recall_thresh:
                    keep_crop = False
            else:
                keep_crop = False
        crop_bbox = [w1,h1,d1,w1 + self.output_size[0], h1 + self.output_size[1], d1 + self.output_size[2]]
        sample['bbox_recall'] = self.recall(crop_bbox, sample['real_bbox'])
        sample['real_bbox'] -= np.array([[w1, h1, d1]])
        sample['real_bbox'][sample['real_bbox']<0] = 0
        sample['real_bbox'] = np.int16(sample['real_bbox'])
        sample['bbox_mask']=np.zeros(self.output_size)[np.newaxis]
        sample['bbox_mask'][:, sample['real_bbox'][0,0]:sample['real_bbox'][1,0], sample['real_bbox'][0,1]:sample['real_bbox'][1,1], sample['real_bbox'][0,2]:sample['real_bbox'][1,2]]=1
        old_start_point = np.array([w1,h1,d1])
        start_point = np.maximum(old_start_point, 0)
        start_pad = start_point-old_start_point
        old_end_point = old_start_point+self.output_size
        end_pad = np.maximum(old_end_point-np.array((w-1, h-1, d-1)), 0)
        end_point = old_end_point - end_pad
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key]=sample[key][:, start_point[0]:end_point[0], \
                    start_point[1]:end_point[1], start_point[2]:end_point[2]]
                sample[key]=F.pad(sample[key], (start_pad[2], end_pad[2], start_pad[1], \
                        end_pad[1], start_pad[0], end_pad[0], 0, 0), mode='constant', value=0)
        return sample

class ExpandLabel(object):
    def __init__(self, kernel_size=[0,0,0], padding=[0,0,0], include=['image']):
        self.kernel = torch.ones(kernel_size, dtype=torch.float32)
        self.padding = padding
        self.include = include

    def __call__(self, sample):
        for key in self.include:
            scribble = sample[key][1:].float()
            scribble = F.conv3d(scribble, self.kernel, padding=self.padding)
            scribble = torch.clamp(scribble, 0, 1).to(torch.uint8)
            sample[key][1:] = scribble
            sample[key][0] = torch.clamp(1-scribble.sum(dim=0), 0, 1)
        return sample

class RandomNoise(object):
    def __init__(self, mean=0, std=0.1,include=['image'], prob=0):
        self.prob = prob
        self.add_noise = tio.RandomNoise(mean=mean, std=std, include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample= self.add_noise(sample)
        return sample


class RandomFlip(object):
    def __init__(self, include=['image'], axes=[], prob=0):
        self.flip_probability = prob
        self.include = include
        self.axes = axes
    def __call__(self, sample):
        axes = random.choice(self.axes)
        flip = tio.RandomFlip(axes=axes, flip_probability=self.flip_probability, include = self.include)
        sample= flip(sample)
        return sample

class RandomTranspose(object):
    def __init__(self, include=['image'], prob=0):
        self.prob = prob
        self.include = include
    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            dim = [1,2,3]
            random.shuffle(dim)
            for key in self.include:
                sample[key] = sample[key].permute(0, dim[0], dim[1], dim[2])
        return sample

class RandomAffine(object):
    def __init__(self, scales=[0.2,0.2,0.2], degrees=[10,10,10],
                include=['image','label'], prob=0):
        self.prob = prob
        self.add_elas = tio.RandomAffine(
            scales=scales,
            degrees=degrees,
            include=include)
        self.include = include

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            for key in self.include:
                sample[key] = sample[key].float()
            sample=self.add_elas(sample)
        return sample
    
class RandomAnisotropy(object):
    def __init__(self, axes=[0], downsampling=(1, 5),
                include=['image'], prob=0):
        self.prob = prob
        self.add_anis = tio.RandomAnisotropy(
            axes=axes,
            downsampling=downsampling,
            include=include)
        self.include = include

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            for key in self.include:
                sample[key] = sample[key].float()
            sample=self.add_anis(sample)
        return sample

class RandomSpike(object):
    def __init__(self, num_spikes=3, intensity=1.2,include=['image'], prob=0):
        self.prob = prob
        self.add_spike = tio.RandomSpike(num_spikes=num_spikes, intensity=intensity,include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_spike(sample)
        return sample

class RandomGhosting(object):
    def __init__(self, intensity=0.8, include=['image'], prob=0):
        self.prob = prob
        self.add_ghost = tio.RandomGhosting(intensity=intensity, include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_ghost(sample)
        return sample

class RandomElasticDeformation(object):
    def __init__(self, num_control_points=[5,10,10], max_displacement=[7,7,7], include=['image','label'], prob=0):
        self.prob = prob
        self.add_elas = tio.RandomElasticDeformation(
            num_control_points=num_control_points,
            max_displacement = max_displacement,
            locked_borders=2,
            image_interpolation='linear',
            label_interpolation='nearest',
            include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_elas(sample)
        return sample

class TargetGenerated(object):
    def __init__(self, key='label', include_class=[]):
        self.key = key
        self.include_class = include_class
    def __call__(self, sample):
        label = sample[self.key].squeeze().numpy().astype(np.int8)
        # del values not in self.include_class
        if len(self.include_class) > 0:
            label[np.logical_not(np.isin(label, self.include_class))] = 0
        labeled_array, num_features = ndimage.label(label)
        slices = ndimage.find_objects(labeled_array)
        zl, xl, yl = label.shape[-3], label.shape[-2], label.shape[-1]
        labels = []
        
        for slice_ in slices:
            if slice_ is None:
                continue
            # 提取该连通区域的类别值
            region_values = label[slice_].flatten()
            # 计算每个类别的出现次数，并找到出现次数最多的类别
            class_id = np.argmax(np.bincount(region_values))
            
            z, x, y = slice_
            dz, dx, dy = z.stop - z.start, x.stop - x.start, y.stop - y.start
            cz, cx, cy = z.start + dz / 2, x.start + dx / 2, y.start + dy / 2
            
            # 将计算得到的标签信息添加到输出列表中
            labels.append([class_id, cz/zl, cx/xl, cy/yl, dz/zl, dx/xl, dy/yl])

        # 如果找到了标签，则将列表转换为tensor；否则，创建一个空的tensor
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty((0, 7))
        # labels_out = torch.zeros((labels.shape[0], 8))
        # labels_out[:, 1:] = labels
        sample[self.key+'_bbox'] = labels
        
        return sample

    

class MultiThreshNormalized(object):
    def __init__(self, include=['image'], thresh_ls=[[-1000, -200, 200, 1000]], norm_ls=[[0,0.2,0.8,1]]):
        self.thresh_ls = thresh_ls
        self.norm_ls = norm_ls
        self.include = include

    def __call__(self, sample):
        # sample["HU"] = {}
        for key in self.include:
            image = sample[key]
            assert isinstance(image, torch.Tensor), "Image must be a PyTorch tensor"
            # sample["HU"][key] = str(round(image.mean(dim=[1, 2, 3]).item()))
            for i in range(len(self.thresh_ls)):
                if i ==0:
                    tensor=img_multi_thresh_normalized_torch(image, self.thresh_ls[i], self.norm_ls[i], data_type=torch.float32)
                else:
                    tensor=torch.cat((tensor, img_multi_thresh_normalized_torch(image, self.thresh_ls[i], self.norm_ls[i], data_type=torch.float32)),0)
            image = tensor
            # if nan in image, warning
            if torch.isinf(image).any():
                print('inf in normalize')
            sample[key]=image
        return sample

class Reshape(object):
    def __init__(self, include=['image'], target_shape=[128, 128, 128], keep_dim_shape=[]):
        self.include = include
        self.target_shape = target_shape
        self.keep_dim_shape = keep_dim_shape

    def __call__(self, sample):
        for key in self.include:
            image = sample[key]
            target_shape = self.target_shape.copy()
            
            if len(self.keep_dim_shape) > 0:
                for dim in self.keep_dim_shape:
                    target_shape[dim] = image.shape[dim+1]
            pad = (0, 0, 0)
            d0, h0, w0 = image.shape[1:]
            d, h, w = target_shape
            sample['shape'] = (image.shape[1:], ((d/d0, h/h0, w/w0), pad))
            # 检查数据类型
            if image.dtype in [torch.int16, torch.int32, torch.int64, torch.int8, torch.uint8]:
                # 如果是整数类型，使用最近邻插值
                interpolation_mode = 'nearest'
                image = torch.nn.functional.interpolate(image.unsqueeze(0), size=target_shape, \
                                                        mode=interpolation_mode).squeeze(0)
            else:
                # 如果是其他类型（如浮点数），使用三线性插值
                interpolation_mode = 'trilinear'
                dtype = image.dtype
                image = torch.nn.functional.interpolate(image.float().unsqueeze(0), size=target_shape, \
                                                        mode=interpolation_mode, align_corners=False).squeeze(0)
                image = image.to(dtype)
            sample[key] = image
            
        return sample


class MultiNormalized(object):
    def __init__(self, include=['image'], thresh_ls=[[-1000, 1000]]):
        self.thresh_ls = thresh_ls
        self.include = include

    def __call__(self, sample):
        for key in self.include:
            image = sample[key]
            for i in range(len(self.thresh_ls)):
                if i ==0:
                    tensor=img_normalized_torch(image.float(), downthresh=self.thresh_ls[i][0], upthresh=self.thresh_ls[i][-1], norm=True, thresh=True).half()
                else:
                    tensor=torch.cat((tensor, img_normalized_torch(image.float(), downthresh=self.thresh_ls[i][0], upthresh=self.thresh_ls[i][-1], norm=True, thresh=True)),0).half()
            image = tensor
            sample[key]=image
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, concat_key=[]):
        self.concat_key=concat_key
    def __call__(self, sample):
        _sample = {}
        if 'image' in sample:
            _sample['image'] = sample['image'].float()
            _sample['image_path'] = {'image_path': sample['image_path']}

        if 'label' in sample:
            _sample['label'] = sample['label'][0].long()
        
        if 'coarseg' in sample:
            sample['coarseg'] = sample['coarseg'][0].long()
        
        if 'crop' in sample:
            _sample['crop'] = sample['crop']

        # if 'type' in sample:
        #     json_dic = sample['type']
        #     for key in json_dic.keys():
        #         json_dic_key = json_dic[key][2]
        #         json_dic_key["lesion location"] = totalseg_class[json_dic_key["lesion location"]]
        #         json_dic_key.pop("size")
        #         json_dic_key.pop("CT value")
        #         json_dic_key["CT value"] = "Mean density of the lesion region is approximately " + sample["HU"]["image"] + " HU,"
        #         if json_dic_key["specific features"] == "": json_dic_key.pop("specific features")
        #         json_dic[key] = json_dic_key
        #     sample['text'] = json_dic
        if 'type' in sample:
            json_dic = sample['type']
            all_keys = list(json_dic.keys())
            if all_keys == []:
                raise ValueError('empty type dictionary')
            elif len(all_keys) == 1:
                selected_key = all_keys[0]
            else:
                if 'coarseg' in sample:
                    coarseg_labels = sample['coarseg'].unique().tolist()
                    existing_keys = [key for key in all_keys if int(key) in coarseg_labels]
                    if not existing_keys:
                        raise ValueError('no pairing between type and coarseg')
                    coarseg_tensor = sample['coarseg']
                    label_tensor = _sample['label']
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
            json_dic_key.pop("size")
            match = re.search(r'approximately (-?\d+(\.\d+)?) HU', json_dic_key['CT value'])
            if match:
                ct_value = float(match.group(1))
                json_dic_key['CT value'] = f'{ct_value} HU,'
            # json_dic_key.pop("CT value")
            # json_dic_key["CT value"] = "Mean density of the lesion region is approximately " + sample["HU"]["image"] + " HU,"
            # if json_dic_key["specific features"] == "": json_dic_key.pop("specific features")
            _sample['text'] = json_dic_key
            
            # if 'label' in _sample and 'coarseg' in sample:
            #     coarseg_tensor = sample['coarseg']
            #     label_tensor = _sample['label']
            #     selected_label = int(selected_key)
            #     mask = (coarseg_tensor == selected_label)
            #     label_tensor[~mask] = 0
            #     _sample['label'] = label_tensor.long()
        
        if len(self.concat_key)>0:
            for key in self.concat_key:
                sample['image'] = torch.cat((sample['image'], sample[key].float()), 0)
        return _sample
    
class SegToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, concat_key=[]):
        self.concat_key=concat_key
    def __call__(self, sample):
        _sample = {}
        if 'seg' in sample:
            # unique_labels = np.unique(sample['seg'])
            # unique_labels = unique_labels[unique_labels != 0]
            # num_classes = len(unique_labels)
            # mapping = {0: 0.0}
            # for i, label in enumerate(unique_labels):
            #     mapping[label] = (i + 1) / num_classes
            # _sample['seg'] = torch.from_numpy(np.vectorize(mapping.get)(sample['seg']).astype(np.float32))
            _sample['seg'] = sample['seg'].float() / 104
            _sample['image_path'] = {'image_path': sample['image_path']}

        if 'label' in sample:
            label = torch.where(sample['label'] == 2, 1, 0)
            labeled_array, num_features = ndimage.label(label)
            if num_features > 0:
                sizes = ndimage.sum(label, labeled_array, range(1, num_features + 1))
                max_label = np.argmax(sizes) + 1
                label = (labeled_array == max_label).astype(np.int64)
            else:
                label = np.zeros_like(label, dtype=np.int64)
            _sample['label'] = torch.from_numpy(label).long()
            
        if 'crop' in sample:
            _sample['crop'] = sample['crop']
        
        if 'type' in sample:
            json_dic = sample['type']
            all_keys = list(json_dic.keys())
            if all_keys == []:
                raise ValueError('empty type dictionary')
            elif len(all_keys) == 1:
                selected_key = all_keys[0]
            else:
                if 'coarseg' in sample:
                    coarseg_labels = sample['coarseg'].unique().tolist()
                    existing_keys = [key for key in all_keys if int(key) in coarseg_labels]
                    if not existing_keys:
                        raise ValueError('no pairing between type and coarseg')
                    coarseg_tensor = sample['coarseg']
                    label_tensor = _sample['label']
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
            json_dic_key.pop("size")
            json_dic_key.pop("CT value")
            if json_dic_key["specific features"] == "": json_dic_key.pop("specific features")
            json_dic_key.pop("density variations")#temporary
            json_dic_key.pop("density")#temporary
            _sample['text'] = json_dic_key

            if 'seg' in sample:
                parts = sample['image_path'].split('/')
                organ = parts[-4].lower()
                disease = parts[-3].lower()

                seg_tensor = sample['seg'].long()
                if 'lung' in organ or 'pneumon' in organ:
                    combined_seg = torch.zeros_like(seg_tensor)
                    for key in all_keys:
                        key_int = int(key)
                        combined_seg = combined_seg | (seg_tensor == key_int)
                    _sample['target'] = combined_seg.long()
                else:
                    selected_label = int(selected_key)
                    seg_tensor = (seg_tensor == selected_label)
                    _sample['target'] = seg_tensor.long()
                
                if 'coarseg' in sample:
                    organs_to_check = ['colon', 'bladder', 'esophagus', 'gall', 'stomach']
                    if any(organ_substring in organ for organ_substring in organs_to_check):
                        if 'stone' in disease:
                            _sample['text']['cavity'] = "within the lumen of a hollow organ,"
                        else:
                            _sample['text']['cavity'] = "wall thickening,"
                    else:
                        coarseg_tensor = sample['coarseg']
                        selected_label = int(selected_key)
                        coarseg_tensor = (coarseg_tensor == selected_label).long()
                        if torch.sum(coarseg_tensor) - torch.sum(_sample['target']) > 100:
                            _sample['text']['cavity'] = "within the parenchymal organ,"
                        else:
                            _sample['text']['cavity'] = "protruding from the parenchymal organ,"
                    _sample['title'] = {'title': str(organ) + ' ' + str(disease) + ', ' + _sample['text']['cavity']}
                    
        return _sample

class TotalToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, max_mask_num=3, crop_size=[], concat_key=[]):
        self.max_mask_num = max_mask_num
        self.crop_size = crop_size
        self.concat_key=concat_key
    def __call__(self, sample):
        _sample = {}
        _sample['max_mask_num'] = {'max_mask_num': self.max_mask_num}
        _sample['crop_size'] = {'crop_size': self.crop_size}
        _sample['image_path'] = {'image_path': sample['image_path']}

        if 'label' in sample:
            _sample['label'] = sample['label'].long()
            unique_labels_check = torch.unique(_sample['label'])
            if 2 in unique_labels_check:
                label = (_sample['label'] == 2)
            elif 1 in unique_labels_check:
                label = (_sample['label'] == 1)
            else:
                raise ValueError(f"No valid labels (1 or 2) found in {_sample['image_path']['image_path']}")

        if torch.sum(label) == 0:
            raise ValueError(f"Empty label in {_sample['image_path']['image_path']}")            
        if 'seg' in sample:
            _sample['seg'] = sample['seg'].long()
        elif 'image' in sample:
            _sample['image'] = sample['image'].half()
        
        if 'coarseg' in sample:
            _sample['coarseg'] = sample['coarseg'].long()

        if 'type' in sample:
            if sample['type'] is {}: raise ValueError(f"Empty dict in {_sample['image_path']['image_path']}")
            _sample['type'] = sample['type']
        
        if len(self.concat_key)>0:
            for key in self.concat_key:
                sample['image'] = torch.cat((sample['image'], sample[key].float()), 0)
        return _sample

def custom_collate_fn(batch):
    """
    自定义批处理函数，自适应处理特定键的数据。
    """
    processed_batch = {}
    # 初始化一个字典，用于存储每个键对应的数据列表
    batch_dict = {key: [] for key in batch[0].keys()}
    
    for sample in batch:
        for key, value in sample.items():
            # 对每个样本的每个键值对，将值添加到对应的列表中
            batch_dict[key].append(value)
    
    for key, value_list in batch_dict.items():
        if 'bbox' in key:  # 对'bboxes'键进行特殊处理
            bboxes = []
            for idx, bboxes_sample in enumerate(value_list):
                if bboxes_sample.nelement() != 0:
                    batch_idx = torch.full((bboxes_sample.size(0), 1), idx, dtype=torch.float32)
                    bboxes_sample = torch.cat((batch_idx, bboxes_sample), dim=1)
                    bboxes.append(bboxes_sample)
            if bboxes:
                processed_batch[key] = torch.cat(bboxes, dim=0)
            else:
                processed_batch[key] = torch.empty((0, 8))
        else:
            # 对其他键使用默认的堆叠方法
            try:
                processed_batch[key] = torch.stack(value_list, dim=0)
            except TypeError:
                # 如果当前键对应的值不支持堆叠（例如字符串等），则保留为列表
                processed_batch[key] = value_list
    
    return processed_batch

def find_and_keep_largest_components(data, target_value, num_to_keep=3):
    """
    找到NIfTI数据中所有标记为目标值的单连通区域，只保留最大的num_to_keep个，
    其他区域赋值为1。

    :param data: NIfTI图像数据数组
    :param target_value: 目标值（例如2或3）
    :param num_to_keep: 保留的最大连通区域数量，默认为3
    :return: 修改后的数据数组
    """
    # 提取标记为目标值的区域
    data[data == 2] = target_value
    binary_data = (data == target_value).astype(np.int8)
    
    # 标记连通区域
    labeled_array, num_features = ndimage.label(binary_data)
    
    if num_features == 0:
        print("No connected components found.")
        return data
    
    # 计算每个连通区域的体积
    sizes = np.array(ndimage.sum(binary_data, labeled_array, range(1, num_features + 1)))
    
    # 找到最大的num_to_keep个连通区域
    largest_indices = np.argsort(sizes)[-num_to_keep:]
    largest_labels = np.array(range(1, num_features + 1))[largest_indices]
    
    # 创建一个新的数据数组，将非最大连通区域赋值为1
    new_data = np.copy(data)
    for alabel in range(1, num_features + 1):
        if alabel not in largest_labels:
            new_data[labeled_array == alabel] = 1
    
    return new_data

def process_label_nii(label_file_path, target_value, num_to_keep=3):
    """
    处理标签NIfTI文件，找到所有目标值的单连通区域，只保留最大的num_to_keep个，
    其他区域赋值为1，并保存修改后的文件。

    :param label_file_path: 标签NIfTI文件路径
    :param target_value: 目标值（例如2或3）
    :param num_to_keep: 保留的最大连通区域数量，默认为3
    :return: 修改后的SimpleITK图像对象
    """
    # 加载NIfTI文件
    label_sitk = sitk.ReadImage(label_file_path)
    label_array = sitk.GetArrayFromImage(label_sitk)
    
    # 处理连通区域
    new_label_array = find_and_keep_largest_components(label_array, target_value, num_to_keep)
    
    # 创建新的SimpleITK图像对象并复制原图像的信息
    new_label_sitk = sitk.GetImageFromArray(new_label_array)
    new_label_sitk.CopyInformation(label_sitk)
    
    return new_label_sitk

def save_to_nnunet(image_array, dirname):
    base = '/ailab/public/pjlab-smarthealth03/leiwenhui/jzq/code/nnUNet/'
    # 检查 image_array 是否只有一个取值
    if np.unique(image_array).size == 1:
        print(f"Image array in {dirname} has only one unique value. Skipping.")
        return

    # 从全局变量字典 nnunet_class 中获取保存路径
    base_path = os.path.dirname(dirname)
    kidneycyst = False
    if base_path == '/ailab/public/pjlab-smarthealth03/leiwenhui/Synlesion/Kidney/Cyst':
        kidneycyst = True
    base_path = base + nnunet_class[base_path]

    # 获取样本名
    sample_name = os.path.basename(dirname)

    # 构建保存路径
    images_tr_dir = os.path.join(base_path, 'imagesTr')
    labels_tr_dir = os.path.join(base_path, 'labelsTr')
    
    # 确保目标目录存在
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)

    # 保存 image_array 到 imagesTr 文件夹
    prefix = 'cyst' if kidneycyst else ''
    image_dest_path = os.path.join(images_tr_dir, f"{prefix}{sample_name}.nii.gz")
    image_sitk = sitk.GetImageFromArray(image_array)
    sitk.WriteImage(image_sitk, image_dest_path)

    # 处理并保存 label.nii.gz 到 labelsTr 文件夹
    label_file_path = os.path.join(dirname, 'label.nii.gz')
    if not os.path.isfile(label_file_path):
        raise FileNotFoundError(f"Label file {label_file_path} does not exist")

    target_value = 3 if kidneycyst else 2
    new_label_sitk = process_label_nii(label_file_path, target_value)
    
    label_dest_path = os.path.join(labels_tr_dir, f"{prefix}{sample_name}.nii.gz")
    sitk.WriteImage(new_label_sitk, label_dest_path)
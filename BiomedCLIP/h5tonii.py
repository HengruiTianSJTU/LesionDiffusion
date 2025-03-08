import SimpleITK as sitk
import h5py
import numpy as np
#dataset_path = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/code/ldm/ldmweek12/dataset/data/'
dataset_path = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/data/BraTS2021/data/'
test_imgs = [
    #'synBraTS2021_01569+BraTS2021_01397.h5'
    'BraTS2021_01569.h5'
]
h5_path = dataset_path + test_imgs[0]
nii_path = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/code/BiomedCLIP/input.nii.gz'

# with h5py.File(h5_path, 'r') as hf:
#     data = hf['samples']
#     data = data[0,:,:,:]
#     img = sitk.GetImageFromArray(data)
#     # 保存为NIfTI格式
#     sitk.WriteImage(img, nii_path)
# print('Already converted')

with h5py.File(h5_path, 'r') as hf:
    data = hf['image']
    data = data[0,:,:,:]
    img = sitk.GetImageFromArray(data)
    # 保存为NIfTI格式
    sitk.WriteImage(img, nii_path)
print('Already converted')
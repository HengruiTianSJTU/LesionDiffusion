from open_clip import create_model_from_pretrained, get_tokenizer, add_model_config
# works on open-clip-torch>=2.23.0, timm>=0.9.8
import torch, numpy
from torchvision.transforms import Resize, ToTensor, Normalize
from urllib.request import urlopen
from PIL import Image
import json
from pathlib import Path
import SimpleITK as sitk
import h5py
import sys
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler

def preprocess_nifti_image(nifti_file_path):
    # read from NIfTI
    itk_img = sitk.ReadImage(str(nifti_file_path))
    img_array = sitk.GetArrayFromImage(itk_img)
    mid_slice = img_array.shape[0] // 2
    img_slice = img_array[mid_slice]
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
    img_slice = img_slice.astype(numpy.uint8)
    pil_img = Image.fromarray(img_slice)
    return pil_img

def preprocess_h5_image(h5_file_path):
    with h5py.File(h5_file_path, 'r') as hf:
        data = hf['image']
        img_array = data[0,:,:,:]
        mid_slice = img_array.shape[0] // 2
        img_slice = img_array[mid_slice]
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
        img_slice = img_slice.astype(numpy.uint8)
        pil_img = Image.fromarray(img_slice)
        return pil_img

acceptible_type = ['png','nii','h5']

if __name__ == "__main__":
# Check if an argument is provided
    if len(sys.argv) < 2:
        input_type = 'png'
    else: input_type = sys.argv[1]# Parse the command line argument
    print(f"Input data type is {input_type}.")

    if input_type not in acceptible_type:
        print(f"Unsupported input type: {input_type}. Please use a type in {acceptible_type}.")
        sys.exit(1)

    pretrained = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/code/BiomedCLIP/open_clip_pytorch_model.bin'
    open_clip_config = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/code/BiomedCLIP/open_clip_config.json'
    tokenizer_config = 'tokenizer_config.json'
    add_model_config(open_clip_config)
    model, preprocess = create_model_from_pretrained(model_name='open_clip_config', pretrained=pretrained)
    tokenizer = get_tokenizer(model_name='open_clip_config')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"running on {device}")
    model.to(device)
    model.eval()

    context_length = 256

    template = 'this is a photo of '
    labels = [
        'Brain Tumor MRI',
        'Brain',
        'Tumor',
        'MRI',
        'Synthetic Brain Tumor MRI'
    ]

    if input_type == 'png':
        dataset_path = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/code/ldm/ldmweek12/dataset/images/test/'
        test_imgs = [
            'input.png',
            'sample.png'
        ]
        imgs_list = test_imgs
        test_imgs = [(Image.open(dataset_path + img)) for img in test_imgs]

    elif input_type == 'nii':
        dataset_path = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/code/ldm/ldmweek12/dataset/samples/'
        test_nifits = [
            'BraTS2021_00022.nii.gz'
        ]
        imgs_list = test_nifits
        test_imgs = [preprocess_nifti_image(dataset_path + nifit) for nifit in test_nifits]

    elif input_type == 'h5':
        dataset_path = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/data/BraTS2021/data/'
        test_h5s = [
            'BraTS2021_01569.h5'
        ]
        imgs_list = test_h5s
        test_imgs = [preprocess_h5_image(dataset_path + nifit) for nifit in test_h5s]

    images = torch.stack([(preprocess(img)) for img in test_imgs]).to(device)
    texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
    with torch.no_grad():
        image_features, text_features, logit_scale = model(images, texts)

        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        logits = logits.cpu().numpy()
        sorted_indices = sorted_indices.cpu().numpy()

    top_k = -1

    for i, img in enumerate(imgs_list):
        pred = labels[sorted_indices[i][0]]

        top_k = len(labels) if top_k == -1 else top_k
        print(img.split('/')[-1] + ':')
        for j in range(top_k):
            jth_index = sorted_indices[i][j]
            print(f'{labels[jth_index]}: {logits[i][jth_index]}')
        print('\n')

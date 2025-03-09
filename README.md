# LesionDiffusion
This repository is for LesionDiffusion, a general, text-controllable lesion synthesis foundation model for 3D CT imaging.

To do list:
- Inpainting pipeline.
- Release public available Dataset.

# Overview
**LesionDiffusion** is a text-controlled framework for generalizable lesion synthesis. The framework consists of two key components: 

1) a lesion mask synthesis network (LMNet) that is guided by lesion bounding boxes (bbox) and mask attributes,
2) a lesion inpainting network (LINet) that is guided by both image attributes and the lesion mask. 

<img src="https://github.com/HengruiTianSJTU/LesionDiffusion/blob/main/fig/Fig2.png" alt="overview" width="900" />

Through extensive experiments, we demonstrate that LesionDiffusion significantly improves segmentation performance across a wide range of lesion types. More importantly, our approach shows exceptional generalization, even for unseen organs and lesion types, achieving a notable improvement in lesion segmentation and surpassing the performance of existing state-of-the-art lesion synthesis models.

<img src="https://github.com/HengruiTianSJTU/LesionDiffusion/blob/main/fig/generation.png" alt="generation quality" width="900" />

In the following sections, we will introduce how to use our LesionDiffusion framework.

# Step -1: Prepare environment. 
Our project was developed using Python 3.11.9, and it is still recommended to use this version for consistency.

1. **Create a Conda environment**:

   First, create a new Conda environment with Python 3.11.9:
   ```bash
   conda create --name LDenv python=3.11.9
   ```
2. **Activate the Conda environment**:

   After creating the environment, activate it using the following commands:
   ```bash
   conda activate LDenv
   ```
3. **Install the required dependencies**: 

   Once the environment is activated, install the necessary dependencies using the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

# Step 0: Prepare dataset
You can download our publicly available dataset from the Hugging Face link provided. If you prefer to use your own data, please ensure it adheres to the instructions outlined below.

### **Key Requirements**  
For every training/validation case, you must provide **four files** with the following content:  

1. **Lesion CT Image**  
   - File format: `.nii.gz`  
   - Content: 3D CT scan containing the lesion.  

2. **Lesion Annotation Mask**  
   - File format: `.nii.gz`  
   - Label definitions:  
     - `0`: Background  
     - `1`: Target organ  
     - `2`: Lesion  

3. **Structured Annotation Report**  
   - File format: `.json`  
   - Content: Structured annotations extracted from radiology reports (e.g., lesion attributes, organ metadata).  

4. **Organ Segmentation Mask**  
   - File format: `.nii.gz`  
   - Content: Segmentation of the organs.  


### **Organizing File Paths**  
To organize your dataset for training and validation, you need to create **eight list files** to map the corresponding file paths. Note that the list filenames do not need to match ours exactly, but you must adhere to the following consistency rules:

- **List Files**:  
  - **Training**:  
    - `train_img_list.txt`: Paths to lesion CT images.  
    - `train_label_list.txt`: Paths to lesion annotation masks.  
    - `train_type_list.txt`: Paths to JSON annotation reports.  
    - `train_seg_list.txt`: Paths to organ segmentation masks.  
  - **Validation**:  
    - `val_img_list.txt`, `val_label_list.txt`, `val_type_list.txt`, `val_seg_list.txt`  

- **Consistency Rules**:  
  1. **Alignment by Row Index**:  
     Entries in all lists **must be aligned by row index**.  
     - For example, line 5 in `train_img_list.txt`, `train_label_list.txt`, and other training lists must correspond to the **same case**.  

  2. **Structured Report Attributes**:  
     Pay special attention to the structured reports in `.json` files. These files must follow a strict dictionary format, as illustrated in the figure below. If you are unsure about the format, refer to the examples provided in the `demo` subdirectory for guidance.  

     <img src="https://github.com/HengruiTianSJTU/LesionDiffusion/blob/main/fig/Fig1.png" alt="attributes" width="900" />
     

# Step 1: Training Pipeline
> **Note**: If you prefer not to train the models from scratch, we provide all required pre-trained weights (`.ckpt` files) on [Google Drive](https://drive.google.com/drive/folders/1n6_eAWhsHBFBYtDxPtp_FXp_tNHhOCHi). You can download and use them directly.

As described in our paper, the **LesionDiffusion** framework consists of two stages, which need to be trained sequentially. Leveraging the PyTorch-Lightning framework, the training process is simplified to configuring files and executing bash commands. Before proceeding, navigate to the `pipeline` subdirectory:

```bash
cd pipeline
```
And the universial training command is as follows:
```bash
torchrun --nnodes=<nodes> --rdzv-endpoint=localhost:<port number> --nproc_per_node <gpus per node> main.py --base <config file path> -t --name <experiment name> --gpus XX,XX
```
### **Stage I: Training LMNet**
To train LMNet, use the configuration file located at `configs/diffusion/maskdiffusion.yaml`. If you are using your own dataset, adjust the `data` section as follows:

- **`seg_list` Parameter**:  
  This parameter is used for training with synthetic data. If you have paired healthy-synthetic data, provide the path to the organ segmentation of the original healthy CT volumes. This helps distill prior knowledge from teacher models. If you just operate with real lesion CT volumes, set the `seg_list` parameter to the same value as `coarseg_list`, which corresponds to the `train_seg_list` or `val_seg_list` files mentioned earlier.

- **`random_sample` and `iter_num` Parameters**:  
  These parameters work together to enable more training iterations per epoch than the actual dataset size. If `random_sample` is set to `true`, the dataloader will randomly sample a training case from the dataset at each iteration until the specified number of iterations (`iter_num`) is reached for the epoch.

After training LMNet, if you have additional datasets or wish to specialize the model for a specific type of lesion, use the fine-tuning configuration file located at `configs/diffusion/maskdiffusion_ft.yaml`. In this file, set the `ckpt_path` parameter in the `model` section to the path of your pretrained LMNet weights to leverage the pretrained results.

### **Stage II: Training VQ & LINet**
In Stage II, you will first train a VQ-GAN model to compress the CT volumes, followed by training the latent-diffusion LINet model using the compressed representations.

#### **Training VQ-GAN**
To train the VQ-GAN model, use the configuration file located at `configs/autoencoder/leiondiffusion_vq.yaml`. If you need to adjust the configuration, ensure that the `n_embed` and `n_classes` parameters in the `model` section remain the same, as they define the size of the codebook used for compression.

#### **Training LINet**
For training the LINet model, use the configuration file located at `configs/latent-diffusion/leiondiffusion.yaml`. The `random_sample` and `iter_num` parameters follow the same requirements as described in Stage I. Additionally, you can fine-tune a pretrained LINet model using the fine-tuning configuration file located at `configs/latent-diffusion/lesiondiffusion_ft.yaml`.

# Step 2: Inference pipeline

# Acknowledgement
- We thank the authors of [ldm](https://github.com/OvO1111/ldm), [PASTA](https://github.com/LWHYC/PASTA), [StableDiffusion](https://github.com/CompVis/latent-diffusion), [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), [VQ-GAN](https://github.com/CompVis/taming-transformers), [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), [nnUNet](https://github.com/MIC-DKFZ/nnUNet), for their great works. Please cite their papers if you use our code.

```bibtex
@misc{lei2025dataefficientpantumorfoundationmodel,
      title={A Data-Efficient Pan-Tumor Foundation Model for Oncology CT Interpretation}, 
      author={Wenhui Lei and Hanyu Chen and Zitian Zhang and Luyang Luo and Qiong Xiao and Yannian Gu and Peng Gao and Yankai Jiang and Ci Wang and Guangtao Wu and Tongjia Xu and Yingjie Zhang and Xiaofan Zhang and Pranav Rajpurkar and Shaoting Zhang and Zhenning Wang},
      year={2025},
      eprint={2502.06171},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2502.06171}, 
}

@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{zhang2024biomedclip,
  title={A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image–Text Pairs},
  author={Sheng Zhang and Yanbo Xu and Naoto Usuyama and Hanwen Xu and Jaspreet Bagga and Robert Tinn and Sam Preston and Rajesh Rao and Mu Wei and Naveen Valluri and Cliff Wong and Andrea Tupini and Yu Wang and Matt Mazzola and Swadheen Shukla and Lars Liden and Jianfeng Gao and Angela Crabtree and Brian Piening and Carlo Bifulco and Matthew P. Lungren and Tristan Naumann and Sheng Wang and Hoifung Poon},
  journal={NEJM AI},
  year={2024},
  volume={2},
  number={1},
  doi={10.1056/AIoa2400640},
  url={https://ai.nejm.org/doi/full/10.1056/AIoa2400640}
}

@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```
# [MICCAI'25 Early Accept] LesionDiffusion
<div align="center">
<br>
<h3>LesionDiffusion: Towards Text-controlled General Lesion Synthesis</h3>

[Hengrui Tian](https://github.com/HengruiTianSJTU/)<sup>1&#42;</sup>&nbsp;
[Wenhui Lei](https://lwhyc.github.io/)<sup>1,2&#42;</sup>&nbsp;
[Linrui Dai](https://github.com/OvO1111/)<sup>1,2&#42;</sup>&nbsp;
Hanyu Chen<sup>3</sup>&nbsp;
[Xiaofan Zhang](https://zhangxiaofan101.github.io/)<sup>1,2</sup>&nbsp;

<sup>1</sup> Shanghai Jiao Tong University &nbsp; <sup>2</sup> Shanghai Artificial Intelligence Laboratory &nbsp; <sup>3</sup> The First Hospital of China Medical University &nbsp;
 
[![ArXiv](https://img.shields.io/badge/ArXiv-<2503.00741>-<COLOR>.svg)](https://arxiv.org/pdf/2503.00741) 

[\[📖 English\]](README.md)
  
</div>

# 概述
**LesionDiffusion** 是一个用于可泛化病灶合成的文本控制框架。该框架由两个关键组件构成：

1) 一个由病灶边界框 (bbox) 和掩码属性引导的病灶掩码合成网络 (LMNet)。
2) 一个由图像属性和病灶掩码共同引导的病灶修复网络 (LINet)。

<img src="https://github.com/HengruiTianSJTU/LesionDiffusion/blob/main/fig/Fig2.png" alt="overview" width="900" />

通过大量的实验，我们证明 LesionDiffusion 显著提升了多种病灶类型的分割性能。更重要的是，我们的方法展现了出色的泛化能力，即使对于未曾见过的器官和病灶类型，也能在病灶分割方面取得显著改进，并超越了现有最先进的病灶合成模型的性能。

<img src="https://github.com/HengruiTianSJTU/LesionDiffusion/blob/main/fig/generation.png" alt="generation quality" width="900" />

接下来的章节将介绍如何使用我们的 LesionDiffusion 框架。本项目基于 [ldm](https://github.com/OvO1111/ldm) 搭建，ldm 将 Stable Diffusion 模型扩展到了三维空间（3D Latent Diffusion Model）。如需了解更多细节，推荐参考原始仓库。

# 第 -1 步：准备环境
我们的项目使用 Python 3.11.9 开发，为了保证一致性，我们仍推荐您使用此版本。

1. **创建 Conda 环境**：

   首先，使用 Python 3.11.9 创建一个新的 Conda 环境：
   ```bash
   conda create --name LDenv python=3.11.9
   ```
2. **激活 Conda 环境**：

   创建环境后，使用以下命令激活它：
   ```bash
   conda activate LDenv
   ```
3. **安装所需依赖**： 

   环境激活后，使用 `requirements.txt` 文件安装必要的依赖：
   ```bash
   pip install -r requirements.txt
   ```

# 第 0 步：准备数据集
您可以从我们提供的 Hugging Face 链接下载公开可用的数据集。如果您希望使用自己的数据，请确保其遵循下述说明。

### **关键要求**  
对于每一个训练/验证案例，您必须提供**四个文件**，内容如下：  

1. **病灶 CT 图像**  
   - 文件格式：`.nii.gz`  
   - 内容：包含病灶的 3D CT 扫描。  

2. **病灶标注掩码**  
   - 文件格式：`.nii.gz`  
   - 标签定义：  
     - `0`: 背景  
     - `1`: 目标器官  
     - `2`: 病灶  

3. **结构化标注报告**  
   - 文件格式：`.json`  
   - 内容：从放射学报告中提取的结构化标注（例如，病灶属性、器官元数据）。  

4. **器官分割掩码**  
   - 文件格式：`.nii.gz`  
   - 内容：器官的分割结果。  


### **组织文件路径**  
为了组织用于训练和验证的数据集，您需要创建**八个列表文件**来映射相应的文件路径。请注意，列表文件名不必与我们的完全一致，但您必须遵守以下一致性规则：

- **列表文件**：  
  - **训练集**：  
    - `train_img_list.txt`: 病灶 CT 图像的路径。  
    - `train_label_list.txt`: 病灶标注掩码的路径。  
    - `train_type_list.txt`: JSON 标注报告的路径。  
    - `train_seg_list.txt`: 器官分割掩码的路径。  
  - **验证集**：  
    - `val_img_list.txt`, `val_label_list.txt`, `val_type_list.txt`, `val_seg_list.txt`  

- **一致性规则**：  
  1. **按行索引对齐**：  
     所有列表中的条目**必须按行索引对齐**。  
     - 例如，`train_img_list.txt`、`train_label_list.txt` 及其他训练列表中的第 5 行必须对应于**同一个病例**。  

  2. **结构化报告属性**：  
     请特别注意 `.json` 文件中的结构化报告。这些文件必须遵循严格的字典格式，如下图所示。如果您不确定格式，请参考 `demo` 子目录中提供的示例以获取指导。  

     <img src="https://github.com/HengruiTianSJTU/LesionDiffusion/blob/main/fig/Fig1.png" alt="attributes" width="900" />
     

# 第 1 步：训练流程
> **注意**：如果您不想从头开始训练模型，我们在 [Google Drive](https://drive.google.com/drive/folders/1n6_eAWhsHBFBYtDxPtp_FXp_tNHhOCHi) 上提供了所有需要的预训练权重（`.ckpt` 文件）。您可以直接下载并使用它们。

正如我们在论文中所述，**LesionDiffusion** 框架包含两个阶段，需要依次进行训练。借助 PyTorch-Lightning 框架，训练过程被简化为修改配置文件和执行 bash 命令。在继续之前，请先进入 `pipeline` 子目录：

```bash
cd pipeline
```
通用的训练命令如下：
```bash
torchrun --nnodes=<节点数> --rdzv-endpoint=localhost:<端口号> --nproc_per_node <每节点GPU数> main.py --base <配置文件路径> -t --name <实验名称> --gpus XX,XX
```
### **阶段一：训练 LMNet**
要训练 LMNet，请使用位于 `configs/diffusion/maskdiffusion.yaml` 的配置文件。如果您使用自己的数据集，请按如下方式调整 `data` 部分：

- **`seg_list` 参数**：  
  此参数用于通过合成数据进行训练。如果您有成对的健康-合成数据，请提供原始健康 CT 图像的器官分割路径。这有助于从教师模型中蒸馏先验知识。如果您仅使用真实的病灶 CT 图像，请将 `seg_list` 参数设置为与 `coarseg_list` 相同的值，该值对应于前面提到的 `train_seg_list` 或 `val_seg_list` 文件。

- **`random_sample` 和 `iter_num` 参数**：  
  这两个参数协同工作，使得每个 epoch 的训练迭代次数可以多于实际数据集的大小。如果 `random_sample` 设置为 `true`，数据加载器 (dataloader) 将在每次迭代时从数据集中随机采样一个训练案例，直到达到为该 epoch 指定的迭代次数 (`iter_num`)。

训练完 LMNet 后，如果您有额外的数据集或希望为特定类型的病灶微调模型，请使用位于 `configs/diffusion/maskdiffusion_ft.yaml` 的微调配置文件。在此文件中，将 `model` 部分的 `ckpt_path` 参数设置为您预训练的 LMNet 权重路径，以利用预训练的成果。

### **阶段二：训练 VQ 和 LINet**
在阶段二中，您将首先训练一个 VQ-GAN 模型来压缩 CT 图像，然后使用压缩后的表示来训练潜空间扩散 (latent-diffusion) 的 LINet 模型。

#### **训练 VQ-GAN**
要训练 VQ-GAN 模型，请使用位于 `configs/autoencoder/lesiondiffusion_vq.yaml` 的配置文件。如果您需要调整配置，请确保 `model` 部分中的 `n_embed` 和 `n_classes` 参数保持相同，因为它们定义了用于压缩的码本 (codebook) 大小。

#### **训练 LINet**
要训练 LINet 模型，请使用位于 `configs/diffusion/lesiondiffusion.yaml` 的配置文件。`random_sample` 和 `iter_num` 参数的要求与阶段一中所述相同。此外，您可以使用位于 `configs/diffusion/lesiondiffusion_ft.yaml` 的微调配置文件来微调一个预训练的 LINet 模型。

# 第 2 步：推理流程
正如我们在论文中所述，**LesionDiffusion** 框架的推理过程包含三个阶段，必须按顺序执行。与训练流程中的操作类似，请确保您仍然在 `pipeline` 子目录中。

### **预处理**

此阶段通过一个脚本 `preprocess.py` 依次执行以下操作：
1. **CT 图像标准化**：预处理原始 CT 图像，以实现统一的方向和体素间距。
2. **器官分割**：使用 TotalSegmentator 分割目标器官。
3. **LLM 报告生成**：调用 LLM API（需设置您的 OpenAI API 密钥）以生成一个虚构的、结构化的放射学报告。
4. **边界框生成**：根据报告为指定的编辑区域创建一个边界框 (bbox)。

在运行预处理脚本之前，请确保已设置您的 OpenAI API 密钥：
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

我们的框架也支持其他的 LLM API。要适配不同的服务商，只需修改 `preprocess.py` 文件开头的 `llm_url` 和 `llm_model` 参数即可。

然后，执行：
```bash
python preprocess.py filelist exp_name attributes
```
这里：
- `filelist` 是一个文本文件的路径，该文件列出了您希望进行修复的原始 CT 图像。文件中的每一行都是一个原始 CT 图像的路径，该图像位于一个特定的子目录中。此子目录也将包含在后续推理过程中生成的其他结果。
- `exp_name` 是您希望为该实验指定的确切名称。脚本将创建一个以此名称命名的子目录，用于存放输出文件：  
  - `bbox_list.txt` 用于边界框  
  - `type_list.txt` 用于结构化影像学报告  
  - `seg_list.txt` 用于 TotalSegmentator 的结果  
  - `img_list.txt` 其内容与 `filelist` 相同
- `attributes` 是一个文本文件的路径，该文件列出了与每个图像样本相对应的属性。每一行都是一个 JSON 格式的字典，其中至少包含 `organ type`（器官类型）和 `lesion type`（病灶类型）。`organ type` 的值在 `organ_type.json` 中定义，而其他可选属性则参考 `description.json`。

这些文件将用于后续的模型推理。

例如，我们的演示用法如下：
```bash
python preprocess.py ../demo/pre_img_list.txt exp ../demo/pre_attr_list.txt
```

### **使用 LMNet 进行推理**
要使用 LMNet 模型进行推理，请使用位于 `configs/diffusion/maskdiffusion_test.yaml` 的配置文件。修改此配置中的 `data` 部分，以引用您在 `exp_name` 子目录中生成的文件列表，并调整 `max_mask_num` 参数以设置每张图像允许的修复区域的最大数量（即采样迭代次数）。

然后执行：
```bash
python test.py --base configs/diffusion/maskdiffusion_test.yaml --name mask-diff-infer --gpus XX,XX
```

### **使用 LINet 进行推理**
完成前述阶段后，我们最后使用 LINet 模型进行推理。请使用位于 `configs/diffusion/lesiondiffusion_test.yaml` 的配置文件。在此配置中，更新 `data` 部分以引用您在 `exp_name` 子目录中生成的文件列表，操作与 `maskdiffusion_test.yaml` 完全相同。此外，您需要修改 `bbox_list.txt` 文件，将其中的每一行的文件名 `bbox.nii.gz` 替换为 `samples_0.nii.gz`。请确保 `max_mask_num` 参数与 `maskdiffusion_test.yaml` 中设置的值保持一致。

然后执行：
```bash
python test.py --base configs/diffusion/lesiondiffusion_test.yaml --name inp-diff-infer --gpus XX
```

对于 LINet 推理，我们强烈建议在启动脚本时，通过 --gpus 选项指定不连续的 GPU 序号（例如` --gpus 0,2 `）。程序会自动使用这些指定的 GPU 进行模型推理，并尝试在每个指定 GPU 的后续一块 GPU 上执行完整图像的 VQ 处理，从而在单张输入图像上实现多位置采样。

所有步骤完成后，您将在每个原始图像对应的子目录中找到您的修复结果。恭喜！

# 致谢
- 我们感谢 [ldm](https://github.com/OvO1111/ldm)、[PASTA](https://github.com/LWHYC/PASTA)、[StableDiffusion](https://github.com/CompVis/latent-diffusion)、[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)、[VQ-GAN](https://github.com/CompVis/taming-transformers)、[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)、[nnUNet](https://github.com/MIC-DKFZ/nnUNet) 的作者们所做的出色工作。如果您的研究中使用了我们的代码，请引用他们的论文。

# 许可协议

本项目采用 MIT 许可协议，详见 [LICENSE](LICENSE) 文件。项目中使用了通过 pip 安装的第三方依赖，其中部分依赖基于 Apache License 2.0 授权。我们未对这些库进行任何修改。  

欢迎您在 MIT 许可协议的条款下自由使用、修改和分享本项目。

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

@article{Wasserthal_2023,
      title={TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images},
      volume={5},
      ISSN={2638-6100},
      url={http://dx.doi.org/10.1148/ryai.230024},
      DOI={10.1148/ryai.230024},
      number={5},
      journal={Radiology: Artificial Intelligence},
      publisher={Radiological Society of North America (RSNA)},
      author={Wasserthal, Jakob and Breit, Hanns-Christian and Meyer, Manfred T. and Pradella, Maurice and Hinck, Daniel and Sauter, Alexander W. and Heye, Tobias and Boll, Daniel T. and Cyriac, Joshy and Yang, Shan and Bach, Michael and Segeroth, Martin},
      year={2023},
      month=sep 
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

# Uni3DAD
This is not the latest code. I will update the latest version...

This is the implementation of [Uni-3DAD: GAN-Inversion Aided Universal 3D Anomaly Detection on Model-free Products](https://arxiv.org/abs/2408.16201).

If you find our work useful in your research, please consider citing: 
```
@article{LIU2025126665,
title = {Uni-3DAD: Gan-inversion aided universal 3D anomaly detection on model-free products},
journal = {Expert Systems with Applications},
volume = {272},
pages = {126665},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.126665},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425002878},
author = {Jiayu Liu and Shancong Mou and Nathan Gaw and Yinan Wang},
keywords = {Unsupervised learning, 3D point clouds, Anomaly detection}
}
```
## How to use code?
### 1. Environment: 
Linux: 20.04 \
Python: 3.8.15 \
Pytorch: 1.13.1 \
CUDA: 11.7 
 
### 2.  Clone the repo:
```bash
git clone https://github.com/JiayuLiu666/Uni3DAD.git
```

### 3. Pip necessary packages: 
Since this repo is built on 3D-ADS (https://github.com/eliahuhorwitz/3D-ADS), M3DM (https://github.com/nomewang/M3DM), Shape-Inversion (https://github.com/junzhezhang/shape-inversion.git), and Shape-guided (https://github.com/jayliu0313/Shape-Guided.git), please refer to their Github pages for the necessary packages and pre-trained models. Thanks for their contributions. 

### 4. Dataset: 
We use MVTec 3D-AD as our dataset (https://www.mvtec.com/company/research/datasets/mvtec-3d-ad). Please refer to 3D-ADS, M3DM, and Shape-guided for the data preprocessing. \
(We also create our own dataset for missing parts detection based on MVTec 3D-AD; if you need it, please contact us.) 

data_process.ipynb is the code that creates the validation dataset.

### 5. Pre-trained models:
You can use
```
train_dist.py 
```
to train your own GAN models for each category. We also provide the pre-trained models: 

After training is finished, you can use 
```
visual_dist.py
```
to check the results of training.


The GAN models and necessary feature-extractors models should be saved like this structure: 
```
├──Checkpoints
│   ├── best_ckpt
│   │   ├── ckpt_00601.pth
│   ├── ...
│   ├── pointMAE_pretrain.pth
│   ├── ...
├── Common
│   ├── ...
│   ├── ...
├── pretrain_checkpoints
│   ├── bagel.ckpt
│   ├── ...
└── README.md
```
## Run the code
You can use 
```bash
python runner.py --METHOD_NAME BTF+GAN --saved_training "YOUR Directory" ...
```
to run the code. Please refer to config.py in the Generation to change the parameters.

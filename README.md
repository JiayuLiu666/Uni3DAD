# Uni3DAD
This is the implementation of [Uni-3DAD: GAN-Inversion Aided Universal 3D Anomaly Detection on Model-free Products](https://arxiv.org/abs/2408.16201).

If you find our work useful in your research, please consider citing: 
```
@article{liu2024uni,
  title={Uni-3DAD: GAN-Inversion Aided Universal 3D Anomaly Detection on Model-free Products},
  author={Liu, Jiayu and Mou, Shancong and Gaw, Nathan and Wang, Yinan},
  journal={arXiv preprint arXiv:2408.16201},
  year={2024}
}
```
## How to use code?
### 1. Environment: 
Linux 20.04 \
Python 3.8.15 \
Pytorch: 1.13.1 \
CUDA: 11.7 
 
### 2.  Clone the repo:
```bash
git clone https://github.com/JiayuLiu666/Uni3DAD.git
```

### 3. pip necessary packages: 
Since this repo is built on 3D-ADS (https://github.com/eliahuhorwitz/3D-ADS), M3DM (https://github.com/nomewang/M3DM), Shape-Inversion (https://github.com/junzhezhang/shape-inversion.git), and Shape-guided (https://github.com/jayliu0313/Shape-Guided.git), please refer to their Github pages for the necessary packages. Thanks for their contributions. 

```bash
pip install -r requirement.txt
# install knn_cuda
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# install pointnet2_ops_lib
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

### 4. Dataset: 
We use MVTec 3D-AD as our dataset (https://www.mvtec.com/company/research/datasets/mvtec-3d-ad). Please refer to 3D-ADS, M3DM, and Shape-guided for the data preprocessing. \
(We also create our own dataset for missing parts detection based on MVTec 3D-AD; if you need it, please contact us.)

### 5. Pre-trained models:
You can use
```
train_dist.py 
```
to train your own GAN models for each category. We also provide the pre-trained models: 
## Run the code


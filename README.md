# Uni3DAD
This is the implementation of [Uni-3DAD: GAN-Inversion Aided Universal 3D Anomaly Detection on Model-free Products](https://arxiv.org/abs/2408.16201).

If you find our work useful in your research, please consider citing: 
```
@misc{uni3dadliu,
      title={Uni-3DAD: GAN-Inversion Aided Universal 3D Anomaly Detection on Model-free Products}, 
      author={Jiayu Liu and Shancong Mou and Nathan Gaw and Yinan Wang},
      year={2024},
      eprint={2408.16201},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.16201}, 
}
```
## How to use code?
1.  Clone the repo:
```bash
git clone https://github.com/JiayuLiu666/Uni3DAD.git
```

2. pip necessary packages: \
Since this repo is built on 3D-ADS (https://github.com/eliahuhorwitz/3D-ADS), M3DM (https://github.com/nomewang/M3DM), Shape-Inversion (https://github.com/junzhezhang/shape-inversion.git), and Shape-guided (https://github.com/jayliu0313/Shape-Guided.git), please refer to their repo for the necessary packages. Thank for their excellent work.  \

3. Dataset:
We use MVTec 3D-AD for our dataset (https://www.mvtec.com/company/research/datasets/mvtec-3d-ad). For missing parts detection, we create our own dataset based on MVTec 3D-AD, you can download via:


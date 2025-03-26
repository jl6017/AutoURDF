# AutoURDF: Unsupervised Robot Modeling from Point Cloud Frames Using Cluster Registration

## [Project page](https://jl6017.github.io/AutoURDF/) | [Paper](https://arxiv.org/abs/2412.05507)

![Teaser image](assets/intro.svg)

This repository contains the official implementation associated with the paper "AutoURDF: Unsupervised Robot Modeling from Point Cloud Frames Using Cluster Registration".

<!-- ## Pipeline

![Teaser image](assets/pipeline.png) -->

## Run

### Environment
```
# setup a vitural environment
conda create -n autourdf python=3.9
conda activate autourdf
```
We used pytorch3D, pybullet, open3d, pyvista.
#### Prepare
For pytorch and pytroch3D installation, please follow this instruction: [pytorch3D installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
#### On my computer
```
Ubuntu 22.04 & CUDA 12.4
python==3.9
torch==2.4.1
torchvision==0.19.1
pytorch3d==0.7.7
```
#### Other packages

```
pip install -r requirements.txt
```

### Data Collection

Collect point cloud sequences, by default wx200_5, 5 sequences
```
bash scripts/dataset.sh
```

### Train Registration Model

Registration, by default wx200_5, run 5 sequences
```
bash scripts/registration.sh
```

### URDF Results

Output URDF, by default wx200_5, run 1 or 5 sequences, unknown DoF infomation
with 5 sequences, 50 frames
```
python PointCloud/coord_map.py --robot wx200_5 --unknown_dof
```
with only 1 sequence, 10 frames
```
python PointCloud/coord_map.py --robot wx200_5 --end_video 1 --unknown_dof
```

#### Demos

Here's a demo of our results:

<div align="center">
  <img src="assets/results_hq.gif" alt="Demo Results" width="100%">
</div>


## Acknowledgments

We sincerely thank [Changxi Zheng](https://www.cs.columbia.edu/~cxz/) and [Ruoshi Liu](https://ruoshiliu.github.io/) for their invaluable feedback.


## BibTex

```
@article{lin2024autourdf,
  title={AutoURDF: Unsupervised Robot Modeling from Point Cloud Frames Using Cluster Registration},
  author={Lin, Jiong and Zhang, Lechen and Lee, Kwansoo and Ning, Jialong and Goldfeder, Judah and Lipson, Hod},
  journal={arXiv preprint arXiv:2412.05507},
  year={2024}
}
```

## Website Template

Our website template is based on:
- [BundleSDF](https://github.com/bundlesdf/bundlesdf.github.io)
- Which in turn was based on [Nerfies](https://nerfies.github.io/)

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
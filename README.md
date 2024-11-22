# Video2URDF

## Installation
```
conda create -n v2urdf python=3.9
conda activate v2urdf
```
### 1. Pytorch, Pytorch3D
V2URDF is implemented using pytorch and [pytorch3D](https://github.com/facebookresearch/pytorch3d/tree/main).\
For pytroch3D installation, please follow this instruction: [pytorch3D installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

On my computer:
* Ubuntu 22.04
* Python 3.9
* cuda 12.4
* Pytorch 2.4.1

Install pytorch:

```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```
### 2. Other packages
We also used open3d, pybullet ...
```
pip install -r requirements.txt
```

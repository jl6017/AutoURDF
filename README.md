# AutoURDF

## Installation
```
conda create -n autourdf python=3.9
conda activate autourdf
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
TODO

## Example
### 1. Collect point cloud sequences, by default wx200_5, 5 sequences
```
bash dataset.sh
```

### 2. Registration, by default wx200_5, run 5 sequences
```
bash registration.sh
```

### 3. Output URDF, by default wx200_5, run 1 or 5 sequences, given DoF infomation
with 5 sequences, 50 frames
```
python PointCloud/coord_map.py --robot wx200_5
```
with only 1 sequence, 10 frames
```
python PointCloud/coord_map.py --robot wx200_5 --end_video 1
```

### 4. Output URDF, by default wx200_5, run 1 or 5 sequences, unknown DoF infomation
with 5 sequences, 50 frames
```
python PointCloud/coord_map.py --robot wx200_5 --unknown_dof
```
with only 1 sequence, 10 frames
```
python PointCloud/coord_map.py --robot wx200_5 --end_video 1 --unknown_dof
```

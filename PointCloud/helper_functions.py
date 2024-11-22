import torch
import numpy as np

# import roma
# def matrix_to_xyzunitquat_roma(t_matrix):
#     unit_quat = roma.rotmat_to_unitquat(torch.tensor(t_matrix[:3, :3]))
#     xyz = torch.tensor(t_matrix[:3, 3])
#     return torch.cat([xyz, unit_quat])

def save_pc_npz(segment_list, path):
    pc_dir = {}
    for i, pc_np in enumerate(segment_list):
        pc_dir[f'{i}'] = pc_np # keyword is string
        pc_dir[f'{i}'] = pc_np # keyword is string

    np.savez(path, **pc_dir)
    
def load_pc_npz(path):
    pc_npz = np.load(path)
    pc_list = [pc_npz[key] for key in pc_npz.keys()]
    return pc_list

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import torch

def matrix2xyzquant_torch(matrix):
    """
    Convert a 4x4 transformation matrix to a 7D vector (x, y, z, qx, qy, qz, qw)
    """
    translation = torch.tensor(matrix[:3, 3])
    rotation = torch.tensor(matrix[:3, :3])
    quaternion = matrix_to_quaternion(torch.tensor(rotation)).squeeze()
    return torch.cat([translation, quaternion])

def xyzquant2matrix_torch(xyzquat):
    """
    Convert a 7D vector (x, y, z, qx, qy, qz, qw) to a 4x4 transformation matrix
    """
    translation = torch.tensor(xyzquat[:3])
    quaternion = torch.tensor(xyzquat[3:])
    rotation = quaternion_to_matrix(quaternion.unsqueeze(0)).squeeze()
    matrix = torch.eye(4)
    matrix[:3, 3] = translation
    matrix[:3, :3] = rotation
    return matrix

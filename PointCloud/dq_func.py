import torch
from pytorch3d.transforms import quaternion_raw_multiply, quaternion_invert, quaternion_to_matrix, matrix_to_quaternion

def transform_from_rot_trans(R: torch.Tensor, t: torch.Tensor):
    ''' Convert rotation matrix and translation vector to
        rigid transformation matrix

        Parameters
        ----------
        R: torch.Tensor, shape (..., 3, 3)
            rotation matrix
        t: torch.Tensor, shape (..., 3)
            translation vector

        Returns
        -------
        T: torch.Tensor, shape (..., 4, 4)
            rigid transformation matrix
    '''
    assert R.shape[-2:] == (3, 3)
    assert t.shape[-1] == 3

    T = torch.zeros(*R.shape[:-2], 4, 4, device=R.device, dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T

def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """ Compute quaternion conjugate

        Parameters
        ----------
        q: torch.Tensor, shape (..., 4)
            quaternion

        Returns
        -------
        q_conj: torch.Tensor, shape (..., 4)
            quaternion conjugate
    """
    assert q.shape[-1] == 4

    q_conj = torch.concatenate([q[..., :1], -q[..., 1:]], dim=-1)
    return q_conj

def quat_trans_to_dualquat(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """ Convert quaternion and translation vector to dual quaternion
    
        Parameters
        ----------
        q: torch.Tensor, shape (..., 4)
            quaternion
        t: torch.Tensor, shape (..., 3)
            translation vector

        Returns
        -------
        dq: torch.Tensor, shape (..., 8)
            dual quaternion
    """
    assert q.shape[-1] == 4
    assert t.shape[-1] == 3
    
    dq_real = q
    zeros = torch.zeros_like(q[..., :1])
    q_dual = torch.concatenate([zeros, t], dim=-1)
    dq_dual = 0.5 * quaternion_raw_multiply(q_dual, dq_real)
    dq = torch.concatenate([dq_real, dq_dual], dim=-1)
    return dq

def rot_trans_to_dualquat(
    R: torch.Tensor, 
    t: torch.Tensor
) -> torch.Tensor:
    """ Convert rotation matrix and translation vector to dual quaternion
    
        Parameters
        ----------
        R: torch.Tensor, shape (..., 3, 3)
            rotation matrix
        t: torch.Tensor, shape (..., 3)
            translation vector

        Returns
        -------
        dq: torch.Tensor, shape (..., 8)
            dual quaternion
    """
    assert R.shape[-2:] == (3, 3)
    assert t.shape[-1] == 3
    eps = torch.finfo(R.dtype).eps
    
    q_rot = matrix_to_quaternion(R)
    q_rot_norm = torch.linalg.norm(q_rot, dim=-1, keepdim=True)
    dq_real = q_rot / torch.clamp_min(q_rot_norm, eps)
    dq = quat_trans_to_dualquat(dq_real, t)
    return dq

def transform_to_dualquat(T: torch.Tensor) -> torch.Tensor:
    """ Convert rigid transformation matrix to dual quaternion
    
        Parameters
        ----------
        T: torch.Tensor, shape (..., 4, 4)
            rigid transformation matrix
            
        Returns
        -------
        dq: torch.Tensor, shape (..., 8)
            dual quaternion
    """
    assert T.shape[-2:] == (4, 4)
    
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    
    # Normalize rotation matrix
    # rotation_det = torch.det(R)
    # rotation_sign = torch.sign(rotation_det)
    # normalized_rotation = rotation_sign.unsqueeze(-1).unsqueeze(-1) * R \
    #     / torch.abs(rotation_det).unsqueeze(-1).unsqueeze(-1)
    # return rot_trans_to_dualquat(normalized_rotation, t)
    return rot_trans_to_dualquat(R, t)

def dualquat_to_quat_trans(dq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ Convert dual quaternion to quaternion and translation vector
    
        Parameters
        ----------
        dq: torch.Tensor, shape (..., 8)
            dual quaternion

        Returns
        -------
        q: torch.Tensor, shape (..., 4)
            quaternion
        t: torch.Tensor, shape (..., 3)
            translation vector
    """
    assert dq.shape[-1] == 8

    dq_real, dq_dual = dq[..., :4], dq[..., 4:]
    q = quaternion_raw_multiply(dq_real, dq_dual)
    t = 2 * quaternion_raw_multiply(dq_dual, quaternion_invert(dq_real))
    return q, t[..., 1:]

def dualquat_to_rot_trans(dq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ Convert dual quaternion to rotation matrix and translation vector
    
        Parameters
        ----------
        dq: torch.Tensor, shape (..., 8)
            dual quaternion

        Returns
        -------
        R: torch.Tensor, shape (..., 3, 3)
            rotation matrix
        t: torch.Tensor, shape (..., 3)
            translation vector
    """
    assert dq.shape[-1] == 8

    dq_real, dq_dual = dq[..., :4], dq[..., 4:]
    R = quaternion_to_matrix(dq_real)
    t = 2 * quaternion_raw_multiply(dq_dual, quaternion_invert(dq_real))
    return R, t[..., 1:]

def dualquat_to_transform(dq: torch.Tensor) -> torch.Tensor:
    """ Convert dual quaternion to rigid transformation matrix
    
        Parameters
        ----------
        dq: torch.Tensor, shape (..., 8)
            dual quaternion

        Returns
        -------
        T: torch.Tensor, shape (..., 4, 4)
            rigid transformation matrix
    """
    assert dq.shape[-1] == 8

    R, t = dualquat_to_rot_trans(dq)
    return transform_from_rot_trans(R, t)

def dualquat_multiply(dq1: torch.Tensor, dq2: torch.Tensor) -> torch.Tensor:
    """ Multiply dual quaternions

        Parameters
        ----------
        dq1: torch.Tensor, shape (..., 8)
            dual quaternion
        dq2: torch.Tensor, shape (..., 8)
            dual quaternion

        Returns
        -------
        dq: torch.Tensor, shape (..., 8)
            dual quaternion
    """
    assert dq1.shape[-1] == 8
    assert dq2.shape[-1] == 8

    dq1_real, dq1_dual = dq1[..., :4], dq1[..., 4:]
    dq2_real, dq2_dual = dq2[..., :4], dq2[..., 4:]
    dq_out_real = quaternion_raw_multiply(dq1_real, dq2_real)
    dq_out_dual = quaternion_raw_multiply(dq1_real, dq2_dual) + quaternion_raw_multiply(dq1_dual, dq2_real)
    dq_out = torch.concatenate([dq_out_real, dq_out_dual], dim=-1)
    return dq_out

def dualquat_invert(dq: torch.Tensor) -> torch.Tensor:
    """ Invert dual quaternion

        Parameters
        ----------
        dq: torch.Tensor, shape (..., 8)
            dual quaternion

        Returns
        -------
        dq_inv: torch.Tensor, shape (..., 8)
            dual quaternion
    """
    assert dq.shape[-1] == 8
    eps = torch.finfo(dq.dtype).eps

    dq_real, dq_dual = dq[..., :4], dq[..., 4:]
    dq_real_norm_squared = torch.linalg.norm(dq_real, dim=-1, keepdim=True) ** 2
    dq_real_conj = quaternion_conjugate(dq_real)
    dq_inv_real = dq_real_conj / torch.clamp_min(dq_real_norm_squared, eps)
    dq_dual_normalized = quaternion_conjugate(dq_dual) / torch.clamp_min(dq_real_norm_squared, eps)
    normalized_dot_product = torch.sum(dq_real * dq_dual, dim=-1, keepdim=True) / torch.clamp_min(dq_real_norm_squared, eps) ** 2
    dq_inv_dual = dq_dual_normalized - 2 * dq_real_conj * normalized_dot_product
    return torch.concatenate([dq_inv_real, dq_inv_dual], dim=-1)

def point_to_dualquat(p: torch.Tensor) -> torch.Tensor:
    """ Convert point to dual quaternion(unit quaternion + translation vector).

        Parameters
        ----------
        p: torch.Tensor, shape (..., 3)
            point

        Returns
        -------
        dq: torch.Tensor, shape (..., 8)
            dual quaternion
    """
    assert p.shape[-1] == 3

    unit_quat = torch.zeros(*p.shape[:-1], 4, device=p.device, dtype=p.dtype)
    unit_quat[..., 0] = 1.0
    zeros = torch.zeros_like(p[..., :1])
    dq = torch.concatenate([unit_quat, zeros, p], dim=-1)
    return dq

if __name__ == "__main__":
    # Example usage
    R = torch.eye(3).unsqueeze(0)  # Identity rotation
    t = torch.tensor([[1.0, 2.0, 3.0]])  # Translation vector
    T = transform_from_rot_trans(R, t)
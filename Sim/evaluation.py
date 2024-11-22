import numpy as np
import os
import glob
from sim_data import SimEnv, data_collection
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance
import pybullet as p
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_offset(raw_data_path):
    offset = []
    # the first joint file
    config_files = sorted(glob.glob(raw_data_path + '/*/'))
    for line in open(config_files[0] + '0000/joint_cfg.txt', 'r'):
        offset_angle = float(line.split(':')[-1])
        offset.append(offset_angle)
    offset = np.array(offset)
    return offset

import numpy as np

def joint_error(pos_a, uv_a, pos_b, uv_b):
    """
    Calculate the joint position error (normal distance) and joint direction error (angle in degrees)
    between two lines represented by points and unit vectors.
    
    Parameters:
    pos_a, pos_b: np.ndarray, shape: (3,)
        The starting positions of the two lines.
    uv_a, uv_b: np.ndarray, shape: (3,)
        The unit direction vectors of the two lines.

    Returns:
    pos_error: float
        The normal distance between the two lines.
    dir_error: float
        The angle between the two direction vectors in degrees.
    """
    # Compute the cross product of the direction vectors
    cross_product = np.cross(uv_a, uv_b)
    cross_product_magnitude = np.linalg.norm(cross_product)

    # Compute normal distance (joint position error)
    if cross_product_magnitude == 0:
        # Lines are parallel, compute distance between points along one of the direction vectors
        diff = pos_b - pos_a
        pos_error = np.linalg.norm(np.cross(diff, uv_a))  # Distance from pos_b to line_a
    else:
        # General case: compute shortest distance between two lines
        diff = pos_b - pos_a
        pos_error = np.abs(np.dot(diff, cross_product)) / cross_product_magnitude

    # Compute direction error (angle in degrees)
    dot_product = np.dot(uv_a, uv_b)
    # Clip dot product to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_radians = np.arccos(dot_product)
    dir_error = np.degrees(angle_radians)

    return pos_error, dir_error


def torch_chamfer_distance(p1, p2):
    """
    p1, p2: open3d.geometry.PointCloud
    """
    p1 = np.asarray(p1.points)
    p2 = np.asarray(p2.points)
    p1 = torch.from_numpy(p1).float()
    p2 = torch.from_numpy(p2).float()
    p1 = p1.unsqueeze(0) # shape: (1, N, 3)
    p2 = p2.unsqueeze(0) # shape: (1, N, 3)
    p1 = p1.to(DEVICE)
    p2 = p2.to(DEVICE)
    loss = chamfer_distance(p1, p2, norm=1)
    return loss[0].item()


# def angle2chamfer(
#         pred_urdf_path: str = None,
#         gt_urdf_path: str = None,
#         dof: int = 5,
#         radius: float = 1.5,
#         num_cameras: int = 20,
#         alist_pred: np.ndarray = None,
#         alist_gt: np.ndarray = None,
# ):
#     """pred"""
#     env_pred = SimEnv(
#         urdf_path=pred_urdf_path,
#         dof=dof,
#         radius=radius,
#         num_cameras=num_cameras
#     )
#     _, pcd_pred_init = data_collection(
#         env=env_pred,
#         angle_list=alist_pred,
#     )
#     env_pred.reset() # disconnect

#     """gt"""
#     env_gt = SimEnv(
#         urdf_path=gt_urdf_path,
#         dof=dof,
#         radius=radius,
#         num_cameras=num_cameras
#     )
#     _, pcd_gt_init = data_collection(
#         env=env_gt,
#         angle_list=alist_gt,
#     )
#     env_gt.reset() # disconnect

#     # visualize the initial configuration
#     o3d.visualization.draw_geometries([pcd_pred_init[0], pcd_gt_init[0]])
#     init_dist = torch_chamfer_distance(pcd_pred_init[0], pcd_gt_init[0])
#     print(init_dist)
    


# def urdf_map(
#         pred_urdf_path: str = None,
#         gt_urdf_path: str = None,
#         pix: int = 800,
#         dof: int = 5,
#         radius: float = 1.5,
#         num_cameras: int = 20,
#         gui: bool = True,
#         visualize: bool = True,
#         offset: np.ndarray = None,
#         sim_ori: list = None
# ):
#     """
#     Find the mapping between the joint sequence and directions of the predicted urdf and the ground truth urdf
#     1. algin the initial configuration
#     2. loop through the joints, control the joints one by one
#     3. find minimum chamfer distance, record the joint sequence and directions
#     """
#     ## align the initial configuration
#     a_list_init = np.zeros((1, dof))
#     a_list_init_offset = a_list_init + offset

#     """pred"""
#     env_pred = SimEnv(
#         urdf_path=pred_urdf_path,
#         gui=gui,
#         dof=dof,
#         radius=radius,
#         num_cameras=num_cameras
#     )
#     _, pcd_pred_init = data_collection(
#         env=env_pred,
#         angle_list=a_list_init,
#     )
#     env_pred.reset() # disconnect

#     """gt"""
#     env_gt = SimEnv(
#         urdf_path=gt_urdf_path,
#         gui=gui,
#         dof=dof,
#         radius=radius,
#         num_cameras=num_cameras,
#         base_orientation=sim_ori
#     )
#     _, pcd_gt_init = data_collection(
#         env=env_gt,
#         angle_list=a_list_init_offset,
#     )
#     env_gt.reset() # disconnect

#     # visualize the initial configuration
#     o3d.visualization.draw_geometries([pcd_pred_init[0], pcd_gt_init[0]])
#     init_dist = torch_chamfer_distance(pcd_pred_init[0], pcd_gt_init[0])
#     print(init_dist)

#     # # get the kinematic tree, branch sequence
#     # import yourdfpy
#     # pred_u = yourdfpy.URDF.load(pred_urdf_path)
#     # gt_u = yourdfpy.URDF.load(gt_urdf_path)

#     # # find the joint that connects the base link and the first link
#     # pred_base_link = pred_u.get_base_link()
#     # gt_base_link = gt_u.get_base_link()

    

def compare_joints(
        joint_map: np.ndarray = None,
        pred_urdf_path: str = None,
        gt_urdf_path: str = None,
        offset: np.ndarray = None,
        sim_ori: list = None,
        pred_ori: list = None,
        dof: int = None    
):
    """pred"""    
    client_id = p.connect(p.DIRECT)
    pred_id = p.loadURDF(
        pred_urdf_path, 
        useFixedBase=True, 
        basePosition=[0, 0, 0], 
        baseOrientation=p.getQuaternionFromEuler(pred_ori),
        globalScaling=GOBAL_SCALE
    )
    num_joints = p.getNumJoints(pred_id)
    revolute_joints = []
    # for gt, we need to get the revolute joint id
    for i in range(num_joints):
        joint_info = p.getJointInfo(pred_id, i)
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            revolute_joints.append(i)

    revolute_joints = revolute_joints[:dof]

    print(revolute_joints)
    
    joint_params_pred = []
    # record the link position and orientation in joint sequence
    link_pos_list = []
    link_ori_list = []
    for i in range(num_joints):
        # get link position and orientation
        link_state = p.getLinkState(pred_id, i)
        link_pos, link_ori = link_state[0], link_state[1]
        link_pos_list.append(link_pos)
        link_ori_list.append(link_ori) 
    for i in revolute_joints:
        joint_id = joint_map[i]
        joint_info = p.getJointInfo(pred_id, joint_id)
        joint_uv, p_pos, p_ori, p_id = joint_info[-4:]
        if p_id == -1:
            link_pos, link_ori = p.getBasePositionAndOrientation(pred_id)
        else:
            link_pos, link_ori = link_pos_list[p_id], link_ori_list[p_id]

        # rotation_matrix = np.array(p.getMatrixFromQuaternion(link_ori)).reshape(3, 3)
        rotation_matrix_l = np.array(p.getMatrixFromQuaternion(link_ori)).reshape(3, 3)
        rotation_matrix_j = np.array(p.getMatrixFromQuaternion(p_ori)).reshape(3, 3)
        rotation_matrix_world = rotation_matrix_l @ rotation_matrix_j.T
        joint_uv = np.array(joint_uv)
        joint_uv_world = rotation_matrix_world @ joint_uv # shape: (3,)
        p_pos = np.array(p_pos)
        p_pos_world = rotation_matrix_l @ p_pos

        # plot vector
        joint_pos_world = p_pos_world + np.array(link_pos) # shape: (3,)

        joint_params_pred.append((joint_pos_world, joint_uv_world))

    p.disconnect()

    """gt"""
    client_id = p.connect(p.DIRECT)
    gt_id = p.loadURDF(gt_urdf_path, useFixedBase=True, basePosition=[0, 0, 0], baseOrientation=p.getQuaternionFromEuler(sim_ori))
    num_joints = p.getNumJoints(gt_id)
    revolute_joints = []
    # for gt, we need to get the revolute joint id
    for i in range(num_joints):
        joint_info = p.getJointInfo(gt_id, i)
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            revolute_joints.append(i)

    revolute_joints = revolute_joints[:dof]

    print(revolute_joints)

    # apply offset
    for i, j_id in enumerate(revolute_joints):
        p.resetJointState(gt_id, j_id, offset[i])

    joint_params_gt = []
    # record the link position and orientation in joint sequence
    link_pos_list = []
    link_ori_list = []
    for i in range(num_joints):
        # get link position and orientation
        link_state = p.getLinkState(gt_id, i)
        link_pos, link_ori = link_state[0], link_state[1]
        link_pos_list.append(link_pos)
        link_ori_list.append(link_ori)
    
    for i in revolute_joints:
        joint_info = p.getJointInfo(gt_id, i)
        joint_uv, p_pos, p_ori, p_id = joint_info[-4:]
        if p_id == -1:
            link_pos, link_ori = p.getBasePositionAndOrientation(gt_id)
        else:
            link_pos, link_ori = link_pos_list[p_id], link_ori_list[p_id]

        rotation_matrix_l = np.array(p.getMatrixFromQuaternion(link_ori)).reshape(3, 3)
        rotation_matrix_j = np.array(p.getMatrixFromQuaternion(p_ori)).reshape(3, 3)
        rotation_matrix_world = rotation_matrix_l @ rotation_matrix_j.T

        joint_uv = np.array(joint_uv)
        joint_uv_world = rotation_matrix_world @ joint_uv
        p_pos = np.array(p_pos)
        p_pos_world = rotation_matrix_l @ p_pos

        # plot vector
        joint_pos_world = p_pos_world + np.array(link_pos)

        joint_params_gt.append((joint_pos_world, joint_uv_world))
    
    p.disconnect()


    # compare the two joint sequences
    pos_error_list = []
    dir_error_list = []
    dir_map = []
    for pred, gt in zip(joint_params_pred, joint_params_gt):
        pred_pos, pred_uv = pred
        gt_pos, gt_uv = gt
        pos_error, dir_error = joint_error(pred_pos, pred_uv, gt_pos, gt_uv)
        # print(f"Position error: {pos_error}, Direction error: {dir_error}")
        if dir_error > 90:
            dir_error = 180 - dir_error
            dir_map.append(-1)
        else:
            dir_map.append(1)

        pos_error_list.append(pos_error)
        dir_error_list.append(dir_error)

    return pos_error_list, dir_error_list, dir_map



def evaluation(
        pred_urdf_path: str = None,
        gt_urdf_path: str = None,
        pix: int = 800,
        dof: int = 5,
        radius: float = 1.5,
        num_cameras: int = 20,
        gui: bool = True,
        visualize: bool = True,
        visualize_result: bool = True,
        save_path: str = None,
        offset: np.ndarray = None,
        sim_ori: list = None,
        pred_ori: list = None,
        joint_map: np.ndarray = None,
        direction_map: list = None
):
    """
    1. load predicted urdf and gt urdf with SimEnv
    2. command the robot to move to a random position
    3. collect point cloud data with data_collection
    """
    os.makedirs(save_path+'pred/', exist_ok=True)
    os.makedirs(save_path+'gt/', exist_ok=True)

    a_list = np.random.rand(3, dof) * 2 - 1 # shape: (3, dof), between -1 and 1
    deg_list = np.degrees(a_list) # shape: (3, dof), convert to degrees

    np.savetxt(save_path+'command_rad.txt', a_list)
    np.savetxt(save_path+'command_deg.txt', deg_list)

    direction = np.array(direction_map) # shape: (dof,)
    inv_map = np.empty_like(joint_map)
    inv_map[joint_map] = np.arange(len(joint_map))

    a_list_direction = a_list * direction # shape: (3, dof), change the direction
    a_list_mapped = a_list_direction[:, inv_map] # shape: (3, dof), reorder the joint sequence
    # a_list_mapped = a_list_mapped * (-1*direction) # shape: (3, dof), change the direction

    print(direction)

    a_list_offset = a_list + offset # shape: (3, 5), broadcasting

    """pred"""
    env_pred = SimEnv(
        urdf_path=pred_urdf_path,
        gui=gui,
        dof=dof,
        radius=radius,
        num_cameras=num_cameras,
        global_scale=GOBAL_SCALE,
        base_orientation=pred_ori
    )
    data_collection(
        env=env_pred,
        data_path=save_path+'pred/',
        width=pix,
        height=pix,
        visualize=visualize,
        angle_list=a_list_mapped,
        num_points=10000
    )
    env_pred.reset() # disconnect

    """gt"""
    env_gt = SimEnv(
        urdf_path=gt_urdf_path,
        gui=gui,
        dof=dof,
        radius=radius,
        num_cameras=num_cameras,
        base_orientation=sim_ori
    )
    data_collection(
        env=env_gt,
        data_path=save_path+'gt/',
        width=pix,
        height=pix,
        visualize=visualize,
        angle_list=a_list_offset,
        num_points=10000
    )
    env_gt.reset() # disconnect

    # compare the two data
    pred_list = sorted(glob.glob(save_path+'pred/*/robot.ply'))
    gt_list = sorted(glob.glob(save_path+'gt/*/robot.ply'))
    loss_record = []

    for deg in deg_list:
        plt.figure(figsize=(8, 4))
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        sns.set_context("notebook", font_scale=1.5)  # Increase overall font size
        
        # Plot with thicker line and black text
        sns.lineplot(x=np.arange(dof), y=deg, marker='o', linewidth=4, color="black")  # Thicker line and black color
        
        plt.ylim(-90, 90)  # Set y-axis range
        plt.yticks([-90, 0, 90], fontsize=14, color="black")  # Set y-axis ticks with large font size and black text
        plt.xticks([], color="black")  # Set x-axis ticks to black
        
        plt.xlabel('Motor Command', fontsize=20, color="black")  # Remove x-axis label but set font size and black color
        # plt.ylabel('Degree', fontsize=20, color="black")  # Set y-axis label with large font size and black color
        
        # plt.savefig(save_path + f'{deg[0]:.2f}_{deg[1]:.2f}_{deg[2]:.2f}.png')
        plt.show()

    # for deg in deg_list:
    #     plt.figure(figsize=(6, 4))
    #     plt.style.use('seaborn-v0_8-whitegrid')
    #     sns.set_palette("husl")
    #     sns.lineplot(x=np.arange(dof), y=deg, marker='o')
    #     plt.xlabel('Motors', fontsize=14)
    #     plt.ylabel('Degree', fontsize=14)
    #     plt.ylim(-90, 90)  # Set y-axis range
    #     plt.yticks([-90, 0, 90], fontsize=14)  # Set y-axis ticks
    #     # disable x ticks
    #     plt.xticks([])
    #     # plt.title('Degree Distribution')
    #     # plt.show()
    #     # save png
    #     plt.savefig(save_path + f'{deg[0]:.2f}_{deg[1]:.2f}_{deg[2]:.2f}.png')

    for pred, gt in zip(pred_list, gt_list):
        pred_pcd = o3d.io.read_point_cloud(pred)
        gt_pcd = o3d.io.read_point_cloud(gt)

        ###########################################
        # ICP filter, only for allegro and op3 because of the noise and scale issue
        icp_result = o3d.pipelines.registration.registration_icp(
            pred_pcd, gt_pcd, 0.01, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000)
        )
        pred_pcd.transform(icp_result.transformation)
        ###########################################

        if visualize_result:
            # color
            # pred_pcd.paint_uniform_color([1, 0.706, 0])
            # gt_pcd.paint_uniform_color([0, 0.651, 0.929])
            # gt_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            # o3d.visualization.draw_geometries([pred_pcd])
            # o3d.visualization.draw_geometries([gt_pcd])
            o3d.visualization.draw_geometries([pred_pcd, gt_pcd])

        loss = torch_chamfer_distance(pred_pcd, gt_pcd)
        print(loss)
        loss_record.append(loss)

    np.savetxt(save_path+'loss.txt', loss_record)
    np.savetxt(save_path+'loss_mean_std.txt', (np.mean(loss_record), np.std(loss_record)))

    
if __name__ == "__main__":
    import argparse
    import json
    np.random.seed(2024) # 2025 only for op3, self collision issue
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='bolt')
    parser.add_argument('--pix', type=int, default=800)
    parser.add_argument('--num_cameras', type=int, default=20) # if num_cameras < 20, use fixed theta, phi, else random theta, phi
    parser.add_argument('--num_cameras_eval', type=int, default=20) # if num_cameras < 20, use fixed theta, phi, else random theta, phi
    parser.add_argument('--step_size', type=int, default=4)

    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--vis_sim', action='store_true')
    parser.add_argument('--vis', action='store_true')    
    args = parser.parse_args()
    # get json parameters by robot name
    with open('parameters.json') as f:
        params = json.load(f)
    robot_params = params[args.robot]
    NUM_SEG = robot_params['num_seg']
    raw_data_path = f'data/raw/{args.robot}/{args.step_size}_deg_{args.num_cameras}_cams/'
    # raw_data_path = f'data/raw/{args.robot}/' # only for real data

    offset = load_offset(raw_data_path)
    sim_ori = robot_params['sim_ori']
    pred_ori = robot_params['ori']
    joint_map = np.loadtxt(f'Sim/joint_map/{args.robot}.txt', dtype=int)

    # GOBAL_SCALE = 1.0 # for first 6
    GOBAL_SCALE = 0.2 # for first allegro, op3

    pos_error_list, dir_error_list, dir_map = compare_joints(
        joint_map=joint_map,
        pred_urdf_path=f'data/urdf/{args.robot}_{NUM_SEG}_seg/{args.step_size}_deg_{args.num_cameras}_cams.urdf',
        gt_urdf_path=robot_params['gt'],
        offset=offset,
        sim_ori=sim_ori,
        pred_ori=pred_ori,
        dof=robot_params['dof']
    )
    EVAL_DIR = 'data/evaluation2/'

    print(f"Position error: {pos_error_list}")
    print(f"Direction error: {dir_error_list}")

    evaluation(
        pred_urdf_path=f'data/urdf/{args.robot}_{NUM_SEG}_seg/{args.step_size}_deg_{args.num_cameras}_cams.urdf',
        gt_urdf_path=robot_params['gt'],
        pix=args.pix,
        radius=robot_params['cam_dist'],
        dof=robot_params['dof'],
        num_cameras=args.num_cameras_eval,
        gui=args.gui,
        visualize=args.vis_sim,
        visualize_result=args.vis,
        save_path=EVAL_DIR + f'{args.robot}_{NUM_SEG}_seg/{args.step_size}_deg_{args.num_cameras}_cams/',
        offset=offset,
        sim_ori=sim_ori,
        pred_ori=pred_ori,
        joint_map=joint_map,
        direction_map=dir_map
    )

    pos_error_mean_std = np.mean(pos_error_list), np.std(pos_error_list)
    dir_error_mean_std = np.mean(dir_error_list), np.std(dir_error_list)
    np.savetxt(EVAL_DIR + f'{args.robot}_{NUM_SEG}_seg/{args.step_size}_deg_{args.num_cameras}_cams/pos_mean_std.txt', pos_error_mean_std)
    np.savetxt(EVAL_DIR + f'{args.robot}_{NUM_SEG}_seg/{args.step_size}_deg_{args.num_cameras}_cams/dir_mean_std.txt', dir_error_mean_std)
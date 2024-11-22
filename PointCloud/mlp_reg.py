import torch
import sklearn.cluster as cluster
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
from model_utils import RegMLP, QRegMLP, DQRegMLP, RRegMLP
from helper_functions import save_pc_npz, load_pc_npz
from cluster_icp import Segments, masked_icp
import glob
import argparse
import json
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.loss import chamfer_distance


def train(m, y, model, clusters, stop=200, learning_rate=0.0002, scheduler_patience=5, scheduler_factor=0.7):
    """
    mlp loop
    Args:
    - m: step matrices, (N, 4, 4) tensor
    - y: target point cloud, (M, 3) tensor
    - model: MLP model, RegMLP
    - clusters: cluster point clouds in local frames, list of (M, 3) tensors, len(clusters) = N
    - stop: early stopping patience
    - learning_rate: initial learning rate
    - scheduler_patience: number of epochs to wait before reducing learning rate
    - scheduler_factor: factor to reduce learning rate by
    Returns:
    - pred_pcd_np: predicted point clouds, list of np arrays
    - pred_pcd: predicted point clouds, list of o3d point clouds, visual
    - m2: updated step matrices, (N, 4, 4) tensor
    - min_loss: best loss value achieved
    """
    import matplotlib
    matplotlib.use('Agg')  # Set backend to non-interactive
    import matplotlib.pyplot as plt
    
    # optimizer
    plot_loss = False
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=False
    )
    
    # train
    min_loss = 1000
    best_pcd = None
    best_m = None
    loss_history = [] if plot_loss else None
    lr_history = [] if plot_loss else None
    count = 0  # for early stopping
    
    for epoch in range(300):
        # forward
        m2 = m.clone()
        """position and rotation"""
        if ROT == 'q':
            q = matrix_to_quaternion(m2[:, :3, :3])  # (N, 4)
            input = torch.cat([m2[:, :3, 3], q], dim=1)  # (N, 6)
            t, r = model(input)  # (N, 3), (N, 3)
            rot = quaternion_to_matrix(r)
            m2[:, :3, :3] = rot
            m2[:, :3, 3] = t
        elif ROT == 'rpy':
            rpy = matrix_to_euler_angles(m2[:, :3, :3], "XYZ")
            input = torch.cat([m2[:, :3, 3], rpy], dim=1)
            t, r = model(input)  # (N, 3), (N, 3)
            rot = euler_angles_to_matrix(r, "XYZ")  # (N, 3, 3)
            m2[:, :3, :3] = rot
            m2[:, :3, 3] = t
        elif ROT == 'dq':
            from dq_func import transform_to_dualquat
            dq = transform_to_dualquat(m2)  # (N, 8)
            input = dq
            dq = model(input)  # (N, 8)
            from dq_func import dualquat_to_transform
            m2 = dualquat_to_transform(dq)  # (N, 4, 4)
        elif ROT == '6d':
            r6d = matrix_to_rotation_6d(m2[:, :3, :3])
            input = torch.cat([m2[:, :3, 3], r6d], dim=1)
            t, r = model(input)
            rot = rotation_6d_to_matrix(r)
            m2[:, :3, :3] = rot
            m2[:, :3, 3] = t
            
        pred_pcd_list = calculate_pc(clusters, m2)  # list of (M, 3) tensors
        pred = torch.cat(pred_pcd_list, dim=0).unsqueeze(0)  # (1, M, 3)
        
        loss, _ = chamfer_distance(pred, y.unsqueeze(0), norm=1)
        
        if plot_loss:
            loss_history.append(loss.item())
            lr_history.append(optimizer.param_groups[0]['lr'])
            
        if loss.item() < min_loss:
            min_loss = loss.item()
            best_pcd = pred_pcd_list
            best_m = m2
            count = 0  # Reset early stopping counter when we find a better loss
        else:
            count += 1
            if count > stop:
                print(f'Early stopping triggered after {epoch} epochs')
                break
                
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        scheduler.step(loss)
        
    pred_pcd_np = [pcd.detach().cpu().numpy() for pcd in best_pcd]  # tensor to np array
    pred_pcd = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd)) for pcd in pred_pcd_np]  # np array to o3d point cloud
    
    print('Best Loss:', min_loss)
    
    if plot_loss:
            # Create figure and save to file instead of displaying
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot loss
            ax1.plot(loss_history)
            ax1.set_title('Training Loss History')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            # Plot learning rate
            ax2.plot(lr_history)
            ax2.set_title('Learning Rate History')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_yscale('log')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save the plot instead of displaying it
            plt.savefig('training_history.png')
            plt.close()
            print("Training history plot saved as 'training_history.png'")
        
    return pred_pcd_np, pred_pcd, best_m, min_loss


def calculate_pc(local_clusters, matrices):
    """
    calculate the point clouds in world frame
    Args:
    - local_clusters: list of (M, 3) tensors, the clusters in local frames
    - matrices: (N, 4, 4) tensor, world to center of clusters
    Returns:
    - pcs: list of (M, 3) tensors, the clusters in world frames
    """
    pcs = []
    for i in range(len(local_clusters)):
        T = matrices[i] # (4, 4) tensor
        ic = local_clusters[i] # (M, 3) tensor
        pc = ic@T[:3, :3].T + T[:3, 3] # (M, 3) tensor, rotation and translation, to world frame
        pcs.append(pc)
    return pcs

def resample_cluster(segments, idx, n_clusters, matrices, normal= False, visual=False):
    """
    resample the segments
    given the new center points and the point cloud in the next step
    Args:
    - segments: Segments class
    - idx: current step index
    - n_clusters: number of clusters
    - matrices: updated step matrices, (N, 4, 4) np array, world to center of clusters
    - visual: visualization flag
    Returns:
    - pcd_np_local_list: list of np arrays, the clusters in local frames
    """

    pc = segments.pc_list[idx]
    pc_np = np.asarray(pc.points)
    # xyz = np.array([matrix[:3, 3] for matrix in matrices]) # (N, 3), translation
    xyz = matrices[:, :3, 3] # (N, 3), translation
    if normal:
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pc.orient_normals_consistent_tangent_plane(30)

        # o3d.visualization.draw_geometries([pc], point_show_normal=True)
        # extend xyz with normals
        xyz_normal = np.hstack([xyz, np.zeros((n_clusters, 3))])

        pc_np = np.asarray(pc.points)
            # add normal
        normals = np.array(pc.normals) * 0.5
        pc_normal_np = np.hstack([pc_np, normals])
        new_clusters = cluster.k_means(pc_normal_np, init=xyz_normal, n_clusters=n_clusters, n_init=1)
    else:
        new_clusters = cluster.k_means(pc_np, init=xyz, n_clusters=n_clusters, n_init=1) # cluster in world frame

    pcd_np_local_list = []

    for i in range(n_clusters):
        mask = new_clusters[1] == i
        i_pcd_np = pc_np[mask]
        inv_matrix = np.linalg.inv(matrices[i]) # center of cluster to world
        i_pcd_np_local = inv_matrix @ np.hstack([i_pcd_np, np.ones((i_pcd_np.shape[0], 1))]).T
        i_pcd_np_local = i_pcd_np_local[:3].T

        # i_pcd_np_local = i_pcd_np@inv_matrix[:3, :3].T + inv_matrix[:3, 3]

        pcd_np_local_list.append(i_pcd_np_local) # list of np arrays, the clusters in local frames

    if visual:
        new_pcd = []
        for i in range(n_clusters):
            i_color = np.random.rand(3)
            mask = new_clusters[1] == i
            i_pcd_np = pc_np[mask]
            i_pcd = o3d.geometry.PointCloud()
            i_pcd.points = o3d.utility.Vector3dVector(i_pcd_np)
            i_pcd.paint_uniform_color(i_color)
            new_pcd.append(i_pcd)

            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            coord.transform(matrices[i])
            coord.paint_uniform_color(i_color)
            new_pcd.append(coord)

        o3d.visualization.draw_geometries(new_pcd)

    return pcd_np_local_list


def match(data_dir, idx):
    ## whether is the first step
    save_dir_list = sorted(glob.glob(f'data/part/{ROBOT}_{NUM_SEG}_seg/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/*/'))
    if len(save_dir_list) == 0:
        # first cluster, k-means cluster, kmeans++
        seg0 = Segments(RAW_PATH_LIST[0])
        seg0.k_means_cluster(0, NUM_SEG, NORMAL) # first step clustering
        step_matrices = np.array(seg0.init_matrix_list) # np array, (N, 4, 4), the first step matrices
        step_cluster_np = seg0.init_segment_list
    else:
        # load the first step
        first_dir = save_dir_list[0]
        step_matrices = np.load(first_dir + 'matrix/0000.npy')
        step_cluster_np = load_pc_npz(first_dir + 'cluster/0000.npz')

    ## create save directory
    sub_dir = data_dir.split('/')[-2]
    save_dir = f'data/part/{ROBOT}_{NUM_SEG}_seg/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/{sub_dir}/'
    os.makedirs(save_dir + 'cluster', exist_ok=True)
    os.makedirs(save_dir + 'matrix', exist_ok=True)

    ## save the first step to output directory
    np.save(save_dir + 'matrix/0000.npy', step_matrices) # 
    save_pc_npz(step_cluster_np, save_dir + 'cluster/0000.npz')

    ## to tensor
    ## scale the point cloud
    step_matrices_tensor = torch.tensor(step_matrices, dtype=torch.float32).to(DEVICE)
    step_cluster_tensor = [torch.tensor(step_cluster_np[i], dtype=torch.float32).to(DEVICE) for i in range(NUM_SEG)] # list of tensors, the clusters in COM coordinates
    step_cluster_tensor_init = [torch.tensor(step_cluster_np[i], dtype=torch.float32).to(DEVICE) for i in range(NUM_SEG)] # list of tensors, the clusters in COM coordinates

    start_idx = 0
    seg = Segments(data_dir) 
    v_flag = False
    best_losses = []

    if ROT == 'dq':
        model = DQRegMLP(hidden_dim=512).to(DEVICE)
        model_rf = DQRegMLP(hidden_dim=512).to(DEVICE)
        print("Using DQRegMLP")
    elif ROT == 'q':
        model = QRegMLP(True,hidden_dim=512).to(DEVICE)
        model_rf = QRegMLP(True,hidden_dim=512).to(DEVICE)
        print("Using QRegMLP")
    elif ROT == 'rpy':
        model = RegMLP(6,3).to(DEVICE)
        model_rf = RegMLP(6,3).to(DEVICE)
        print("Using RegMLP")
    elif ROT == '6d':
        model = RRegMLP(hidden_dim=512).to(DEVICE)
        model_rf = RRegMLP(hidden_dim=512).to(DEVICE)
        print("Using RRegMLP")

    for i in tqdm(range(start_idx, seg.data_size - 1, 1)):
        # print(f'Step {i+1}')
        target_pcd_np = np.array(seg.pc_list[i+1].points)
        target_pcd = torch.tensor(target_pcd_np, dtype=torch.float32).to(DEVICE)

        if MLP_ICP:
            """MLP + ICP"""
            ## global
            pred_pcd_np, pred_pcd, step_m, best_loss = train(
                m=step_matrices_tensor,
                y=target_pcd,
                model=model,
                clusters=step_cluster_tensor
            )
            
            best_losses.append(best_loss)

            ## visualize
            if VIS:
                if i > 0 and i % 3 == 0:
                    vis_pcd = []
                    vis_pcd.append(seg.pc_list[i+1])
                    # read the target point cloud
                    o3d.visualization.draw_geometries([*pred_pcd])
                    o3d.visualization.draw_geometries([*pred_pcd, *vis_pcd])
                    v_flag = True
                else:
                    v_flag = False

            step_m_np = step_m.detach().cpu().numpy() # (N, 4, 4)

            # ICP
            world_clusters, matrices = masked_icp(step_cluster_np, pred_pcd_np, target_pcd_np, step_m_np, v_flag, ori=False)
            new_seg_np = resample_cluster(seg, i+1, NUM_SEG, matrices, NORMAL, v_flag) # list of np arrays, the segments in COM coordinates
            step_cluster_tensor = [torch.tensor(new_seg_np[j], dtype=torch.float32).to(DEVICE) for j in range(NUM_SEG)] # update model input clusters
            step_matrices_tensor = torch.tensor(matrices, dtype=torch.float32).to(DEVICE) # update model input matrices

            ## save
            np.save(save_dir + f'matrix/{(i+1):04}.npy', matrices)
            save_pc_npz(new_seg_np, save_dir + f'cluster/{(i+1):04}.npz')
        
        else:
            """MLP + MLP, default version"""
            ## step match
            print('Step')
            pred_pcd_np, pred_pcd, step_m, best_loss = train(
                m=step_matrices_tensor,
                y=target_pcd,
                model=model,
                clusters=step_cluster_tensor # updated every loop step
            )
            step_matrices_tensor = step_m.detach().clone().to(DEVICE) # update model input matrices
            
        
            ## anchor match
            print('Anchor')
            pred_pcd_np, pred_pcd, step_m, best_loss = train(
                m=step_matrices_tensor,
                y=target_pcd,
                model=model_rf,
                clusters=step_cluster_tensor_init, # initilized cluster in step 1
                learning_rate=0.0001
            )
            step_matrices_tensor = step_m.detach().clone().to(DEVICE) # update model input matrices # update model input matrices
            best_losses.append(best_loss)  # only for refine model

            ## visualize
            if VIS:
                if i > -1 and i % 2 == 0:
                    vis_pcd = []
                    vis_pcd.append(seg.pc_list[i+1])
                    # read the target point cloud
                    o3d.visualization.draw_geometries([*pred_pcd])
                    o3d.visualization.draw_geometries([*pred_pcd, *vis_pcd])
                    v_flag = True
                else:
                    v_flag = False

            step_m_np = step_m.detach().cpu().numpy() # (N, 4, 4) np array for resampling
            new_seg_np = resample_cluster(seg, i+1, NUM_SEG, step_m_np, NORMAL, v_flag) # list of np arrays, the segments in COM coordinates
            
            step_cluster_tensor = [torch.tensor(new_seg_np[j], dtype=torch.float32).to(DEVICE) for j in range(NUM_SEG)] # update model input clusters

            ## save
            np.save(save_dir + f'matrix/{(i+1):04}.npy', step_m_np)
            save_pc_npz(new_seg_np, save_dir + f'cluster/{(i+1):04}.npz')

            # save_pc_npz(pred_pcd_np, save_dir + f'cluster/{(i+1):04}.npz') # if no resampling

    #save as txt
    if LOSS:
        np.savetxt(save_dir + 'loss.txt', best_losses)
        # os.makedirs(f'loss/{ROBOT}/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/', exist_ok=True)
        # np.savetxt(f'loss/{ROBOT}/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/{idx}_{ROT}.txt', best_losses)

if __name__ == "__main__":
    ## parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', DEVICE)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='nao')
    parser.add_argument('--mlp_icp', action='store_true')
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--loss', action='store_true')
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--num_cameras', type=int, default=20) # number of cameras
    parser.add_argument('--step_size', type=int, default=4) # motor step size
    parser.add_argument('--num_video', type=int, default=5)
    parser.add_argument('--r', type=str, default='q', choices=['q', 'rpy', 'dq', '6d']) # rotation representation
    args = parser.parse_args()

    # get json parameters by robot name
    with open('parameters.json') as f:
        params = json.load(f)
    robot_params = params[args.robot]

    # json parameters
    ROBOT = args.robot
    NUM_SEG = robot_params['num_seg']
    DOF = robot_params['dof']
    STEP_SZIE = args.step_size
    NUM_CAMERAS = args.num_cameras
    # SAMPLE = args.sample
    MLP_ICP = args.mlp_icp
    VIS = args.visual
    ROT = args.r
    LOSS = args.loss
    NORMAL = args.normal

    RAW_PATH_LIST = sorted(glob.glob(f'data/raw/{ROBOT}/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/*/'))
    if len(RAW_PATH_LIST) == 0: # for real data
        RAW_PATH_LIST = sorted(glob.glob(f'data/raw/{ROBOT}/*/'))
    
    print(f'Found {len(RAW_PATH_LIST)} raw data directories')

    # FIRST_DIR = f'data/part/first/{ROBOT}_{NUM_SEG}/'
    # if len(glob.glob(FIRST_DIR)) == 0:
    #     first_cluster()

    for i, data_dir in enumerate(RAW_PATH_LIST[:args.num_video]):
        match(data_dir, i)
import numpy as np
import open3d as o3d
import glob
import matplotlib.pyplot as plt
import os
from skimage import measure
import scipy.ndimage
import mcubes
import copy
import pyvista as pv
import pymeshfix
from scipy.ndimage import distance_transform_edt
import argparse
import json

from helper_functions import save_pc_npz, load_pc_npz
from cluster_icp import Segments, masked_icp

def voxel_to_sdf(voxel_grid):
    # Calculate the unsigned distance field
    unsigned_sdf = distance_transform_edt(1 - voxel_grid)
    
    # Calculate the signed distance field
    signed_sdf = unsigned_sdf - distance_transform_edt(voxel_grid)
    
    return signed_sdf

def load_cm():
    from coord_map import CoordMap
    sub_path = sorted(glob.glob(PART_PATH + '*/'))

    cm_list = []
    sum_map_list = []
    path_list = []
    for path in sub_path:
        # seed_list.append(int(path.split('V', 1)[1]))

        path_list.append(path.split('/')[-2])

        cm = CoordMap(path, matrix=True, start_steps=START_STEPS, end_steps=END_STEPS)
        _, sum_map = cm.coord_dist_map(diff=False, p_xyz=XYZ_R)
        cm_list.append(cm)
        sum_map_list.append(sum_map)

    sum_map = np.mean(sum_map_list, axis=0)

    cluster_idx, g0, g1 = cm_list[0].coord_cluster(sum_map, num_links=DOF+1) # TODO: function deleted
    edges = list(g1.edges)

    return cm_list, cluster_idx, edges, path_list

# def visualize_links(links):
#     pcd_list = []
#     for link in links:
#         avg_link = np.concatenate(link['clusters'], axis=0) # (N, 3)
#         forward_matrices = link['matrices']

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(avg_link)
#         # transform the point cloud
#         pcd.transform(forward_matrices[50])
#         pcd.paint_uniform_color(np.random.rand(3))
#         pcd_list.append(pcd)

#     o3d.visualization.draw_geometries(pcd_list)


def save_links(cm_list, cluster_idx, path_list, start_steps, end_steps):
    for cm, link_dir in zip(cm_list, path_list):
        # link_dir = link_path + path + '/'
        os.makedirs(link_dir + 'cluster', exist_ok=True)
        os.makedirs(link_dir + 'matrix', exist_ok=True)
        os.makedirs(link_dir + 'cluster_wf', exist_ok=True)
        links_cm = cm.cluster_to_link(cluster_idx)
        time_steps = end_steps - start_steps
        for t in range(time_steps):
            links_matrices_t = [link['matrices'][t] for link in links_cm] # (N, 4, 4)
            links_clusters_t = [link['clusters'][t] for link in links_cm] # (N, num_points, 3)
            links_clusters_wf_t = [link['clusters_wf'][t] for link in links_cm] # (N, num_points, 3)
            np.save(link_dir + f'matrix/{t:04}.npy', links_matrices_t)
            save_pc_npz(links_clusters_t, link_dir + f'cluster/{t:04}.npz')
            save_pc_npz(links_clusters_wf_t, link_dir + f'cluster_wf/{t:04}.npz')


def refine_links_clusters(path_list, start_steps, end_steps, dof):
    """
    match clusters_i to clusters_0, in local frame
    """
    for link_dir in path_list:
        # link_dir = link_path + path + '/'
        # link_m_files = sorted(glob.glob(link_dir + 'matrix/*.npy'))
        link_c_files = sorted(glob.glob(link_dir + 'cluster/*.npz'))
        os.makedirs(link_dir + 'cluster_rf', exist_ok=True)
        for t in range(start_steps, end_steps):
            link_clusters = load_pc_npz(link_c_files[t]) # (N, num_points, 3)
            link_clusters_first = load_pc_npz(link_c_files[start_steps]) # (N, num_points, 3)
            # link_clusters_first_10 = [load_pc_npz(link_c_files[i]) for i in range(5)]

            link_cluster_icp_np_list = []
            # icp
            for link_id, link_cluster, link_cluster_first in zip(range(dof+1), link_clusters, link_clusters_first):
                # combine the first 10 clusters
                # link_cluster_first_10_combined = np.concatenate([c[link_id] for c in link_clusters_first_10] , axis=0) # (N, 3)
                # first_10_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(link_cluster_first_10_combined))
                init_m = np.eye(4)
                link_cluster_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(link_cluster))
                link_cluster_first_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(link_cluster_first))
                # normal estimation
                # link_cluster_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
                # link_cluster_first_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
                threshold = 1

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    link_cluster_pc, link_cluster_first_pc, threshold, init_m,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000)
                )

                rf_matrix = reg_p2p.transformation.copy()

                # rf_matrix = init_m

                link_cluster_icp = link_cluster_pc.transform(rf_matrix)
                link_cluster_icp_np = np.asarray(link_cluster_icp.points) # (num_points, 3)
                link_cluster_icp_np_list.append(link_cluster_icp_np)
            
            save_pc_npz(link_cluster_icp_np_list, link_dir + f'cluster_rf/{t:04}.npz')



def visualize_links(path_list, start_steps, end_steps, dof, vis_flow):
    for link_dir in path_list:
        # link_dir = link_path + path + '/'
        link_m_files = sorted(glob.glob(link_dir + 'matrix/*.npy'))
        link_c_files = sorted(glob.glob(link_dir + 'cluster/*.npz'))
        # link_cwf_files = sorted(glob.glob(link_dir + 'cluster_wf/*.npz'))
        link_mrf_files = sorted(glob.glob(link_dir + 'matrix_rf/*.npy'))
        link_crf_files = sorted(glob.glob(link_dir + 'cluster_rf/*.npz'))

        pcd_list = []
        pcd_rf_list = []
        pcd_local_list = []
        pcd_rf_local_list = []
        for link_id in range(dof+1):
            link_c = [load_pc_npz(link_c_files[i])[link_id] for i in range(start_steps, end_steps)]
            link_crf = [load_pc_npz(link_crf_files[i])[link_id] for i in range(start_steps, end_steps)] # (N, num_points, 3)
            link_c_combined = np.concatenate(link_c, axis=0) # (N, 3)
            link_crf_combined = np.concatenate(link_crf, axis=0) # (N, 3)
            link_m = np.load(link_m_files[0])[link_id] # (4, 4)

            # vis link in world frame

            vis = []
            for t in range(start_steps, end_steps, 1):
                link_crf_i = link_crf[t]
                link_m_i = np.load(link_m_files[t])[link_id]
                pcd_rf_i = o3d.geometry.PointCloud()
                pcd_rf_i.points = o3d.utility.Vector3dVector(link_crf_i)
                # pcd_rf_i.paint_uniform_color(np.random.rand(3))
                pcd_rf_i.transform(link_m_i)

                # coord
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                coord.transform(link_m_i)
                vis.append(pcd_rf_i)
                vis.append(coord)


            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(link_c_combined)
            pcd_local = copy.deepcopy(pcd)
            # pcd.paint_uniform_color(np.random.rand(3))
            pcd.transform(link_m)

            pcd_rf = o3d.geometry.PointCloud()
            pcd_rf.points = o3d.utility.Vector3dVector(link_crf_combined)
            pcd_rf_local = copy.deepcopy(pcd_rf)
            # pcd_rf.paint_uniform_color(np.random.rand(3))
            pcd_rf.transform(link_m)

            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord.transform(link_m)

            if vis_flow:
                o3d.visualization.draw_geometries(vis)
                o3d.visualization.draw_geometries([pcd, pcd_rf, coord])

            pcd_list.append(pcd)
            pcd_rf_list.append(pcd_rf)
            pcd_local_list.append(pcd_local)
            pcd_rf_local_list.append(pcd_rf_local)

        # o3d.visualization.draw_geometries(pcd_rf_list)
        # o3d.visualization.draw_geometries(pcd_list)

        # save the refined clusters as ply
        for i, pcd in enumerate(pcd_rf_local_list):
            o3d.io.write_point_cloud(link_dir + f'{i:04}.ply', pcd)

        for i, pcd in enumerate(pcd_local_list):
            o3d.io.write_point_cloud(link_dir + f'{i:04}_og.ply', pcd)


def link_mesh(path_list, dof, vsize, vis_flow):
    """
    load point clouds from .ply files
    point clouds to voxel grids
    voxel grids to mesh (marching cubes)
    """
    for path in path_list:
        colormap = plt.get_cmap("jet")
        colors = [colormap(i / (dof+1)) for i in range(dof+1)]
        for i in range(dof+1):
            pc_path = path + f'/{i:04}.ply'
            print(pc_path)
            pcd = o3d.io.read_point_cloud(pc_path)
            # clean outliers
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)

            #o3d.visualization.draw_geometries([pcd])
            # downsample to 5000 points
            # pcd = pcd.random_down_sample(0.5)
            voxel_size = vsize # 0.003
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
            #o3d.visualization.draw_geometries([voxel_grid])

            """marching cubes"""
            
            voxel_indices = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
            # print(voxel_indices.shape)

            # Create an empty NumPy volume (3D array)
            min_bound = voxel_grid.get_min_bound()
            max_bound = voxel_grid.get_max_bound()

            # Compute the size of the 3D array based on the voxel grid bounds and voxel size
            volume_size = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
            volume = np.zeros(volume_size, dtype=bool)

            volume = np.zeros(volume_size, dtype=np.uint8)

            # Set occupied voxels to 1
            for idx in voxel_indices:
                volume[idx[0], idx[1], idx[2]] = 1

            # sdf = voxel_to_sdf(volume)
            # print(volume.shape)
            # print(sdf.shape)

            # Marching cubes
            # Define the threshold to extract the surface
            # Typically, this is the level at which you want to generate the surface
            # threshold = 0.001

            # # Apply the marching cubes algorithm
            # verts, faces, normals, values = measure.marching_cubes(volume, level=threshold)

            threshold = 0.

            # Apply marching cubes using PyMCubes
            verts, faces = mcubes.marching_cubes(volume, threshold)

            # scale the mesh
            verts = verts * voxel_size + min_bound

            # Create a mesh from the vertices and faces
            mesh = o3d.geometry.TriangleMesh()

            # Convert the vertices and faces to an Open3D mesh
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)

            # mesh.remove_duplicated_vertices()
            # mesh.remove_duplicated_triangles()
            # mesh.remove_non_manifold_edges()

            # Optionally, you can compute vertex normals
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.3, 0.3, 0.3])

            # Visualize the mesh using Open3D
            #o3d.visualization.draw_geometries([mesh])

            mesh_out = mesh.filter_smooth_simple(number_of_iterations=1)
            # mesh_out = mesh.filter_smooth_laplacian(number_of_iterations=100)

            mesh_out.compute_vertex_normals()
            #o3d.visualization.draw_geometries([mesh_out])

            # Convert the mesh to a PyVista mesh

            vertices = np.asarray(mesh_out.vertices)
            triangles = np.asarray(mesh_out.triangles)
            # vclean, fclean = pymeshfix.clean_from_arrays(vertices, triangles)
            meshfix = pymeshfix.MeshFix(vertices, triangles)
            #meshfix.plot()
            meshfix.repair()
            mesh_pv = meshfix.mesh

            # mesh_pv = pv.PolyData(vertices, np.hstack((np.full((triangles.shape[0], 1), 3), triangles)))

            # # Convert vertices and faces from PyMCubes to a format PyVista can use
            # mesh = pv.PolyData(verts, faces)

            # # Visualize the mesh
            if vis_flow:
                plotter = pv.Plotter()
                # plotter.add_mesh(mesh_pv, color=colors[i])
                plotter.add_mesh(mesh_pv, color='white', show_edges=True)
                plotter.show()

            # # Save the mesh to a file
            mesh_pv.save(path + f'/{i:04}.stl')

            # # Save the mesh to a file
            # o3d.io.write_triangle_mesh(LINK_PATH + f'{(seed):04}/{i:04}.stl', mesh)
        

        
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='wx200_5')
    parser.add_argument('--xyz_r', type=float, default=0.5)
    parser.add_argument('--start_steps', type=int, default=0)
    parser.add_argument('--end_steps', type=int, default=10)
    parser.add_argument('--build_links', type=int, default=1)
    parser.add_argument('--vis_flow', action='store_true')
    parser.add_argument('--num_cameras', type=int, default=3) # number of cameras
    parser.add_argument('--step_size', type=int, default=5) # motor step size
    args = parser.parse_args()

    XYZ_R = args.xyz_r
    START_STEPS = args.start_steps
    END_STEPS = args.end_steps
    BUILD_LINKS = args.build_links
    VIS_FLOW = args.vis_flow
    # get json parameters by robot name
    with open('parameters.json') as f:
        params = json.load(f)
    robot_params = params[args.robot]

    ROBOT = args.robot
    NUM_SEG = robot_params['num_seg']
    DOF = robot_params['dof']
    ORI = robot_params['ori']
    VSIZE = robot_params['voxel_size']
    STEP_SZIE = args.step_size
    NUM_CAMERAS = args.num_cameras

    PART_PATH = f'data/part/{ROBOT}_{NUM_SEG}_seg/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/'
    LINK_PATH = f'data/mesh/{ROBOT}_{NUM_SEG}_seg/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/'


    cm_list, cluster_idx, _, path_list = load_cm()

    path_list = path_list[:BUILD_LINKS]
    sub_link_path = [LINK_PATH + path + '/' for path in path_list]

    save_links(cm_list, cluster_idx, sub_link_path, START_STEPS, END_STEPS) # save clusters and matrices
    refine_links_clusters(sub_link_path, START_STEPS, END_STEPS, DOF) # refine clusters icp
    visualize_links(sub_link_path, START_STEPS, END_STEPS, DOF, VIS_FLOW) # load clusters and matrices, visualize in world frame
    link_mesh(sub_link_path, DOF, VSIZE, VIS_FLOW)  # load clusters, convert to voxel grids, convert to mesh, save mesh


    
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import glob
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score

import networkx as nx
import time
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from compute_joints import estimate_joint_axes_from_tree, create_urdf, visualize_urdf
from visualize import visualize_kinematic_tree
import roma
from link import save_links, refine_links_clusters, visualize_links, link_mesh
from helper_functions import xyzquant2matrix_torch, matrix2xyzquant_torch

import argparse
import json
import os
import math

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def lineset2cylinder(lineset, nodes, scale=1):
    """
    Convert a LineSet to a cylinder and spheres, open3d
    Convert a pcd to spheres, open3d
    """
    # create cylinder between two points
    R_cylinder = 0.01 * scale
    R_sphere = 0.02 * scale

    robograph = []
    for edge in lineset.lines:
        p1 = lineset.points[edge[0]]
        p2 = lineset.points[edge[1]]
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=R_cylinder, height=np.linalg.norm(p1-p2))
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color([0.5, 0.5, 0.5])

        cylinder = cylinder.translate((p1+p2)/2)
        rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), p2-p1)

        cylinder = cylinder.rotate(rotation_matrix, center=(p1+p2)/2)

        robograph.append(cylinder)

    # create spheres
    for node in nodes.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=R_sphere)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.1, 0.1, 0.1])
        sphere = sphere.translate(node)
        robograph.append(sphere)

    return robograph


def coord_clustering(num_coords, d_map, num_links):
    """
    Cluster the coordinates based on the distance variance matrix
    """

    orignal_idx = np.arange(num_coords) 


    threshold = 1 # 0
    while True:
        edges = []
        for i in range(num_coords):
            for j in range(i+1, num_coords):
                if d_map[i, j] < threshold: #  in edges_base and (i, j) in edges_nn
                    # build the graph based on the MST
                    edges.append((i, j))

        G1 = nx.Graph()
        G1.add_nodes_from(orignal_idx)
        G1.add_edges_from(edges)
        cluster_idx = list(nx.connected_components(G1))
        threshold -= 0.0001 # +=

        if len(cluster_idx) >= num_links: # <=
            print("Threshold: ", threshold) 
            break

    # get labels
    cluster_labels = np.full(num_coords, -1) 

    # Assign a unique label for each connected component (cluster)
    for cluster_id, cluster in enumerate(cluster_idx):
        for index in cluster:
            cluster_labels[index] = cluster_id

    # print(cluster_labels)

    silhouette_avg = silhouette_score(d_map, cluster_labels, metric="precomputed")

    print(f"n={num_links}, silhouette score:{silhouette_avg}")

    return cluster_idx, G1, silhouette_avg


def silhouette_score_method(num_coords, d_map, link_range=(3, 15)):
    """
    Silhouette Score Method
    """
    s_score = []
    nls = np.arange(link_range[0], link_range[1])
    for nl in nls:
        cluster_idx, g1, silhouette = coord_clustering(num_coords, d_map, num_links=nl)
        s_score.append(silhouette)
    
    # max silhouette score as the number of links
    num_links = nls[np.argmax(s_score)]

    cluster_idx, g1, _ = coord_clustering(num_coords, d_map, num_links=num_links)

    return cluster_idx, g1, s_score, nls

class CoordMap:
    """
    Class for coordinate correlation map
    - self.coords: (time-step, num_coords, 6) np array, xyzrpy
    - self.matrices: (time-step, num_coords, 4, 4) np array
    - self.clusters: (time-step, num_coords, num_points, 3) list of list of np arrays
    """
    def __init__(self, data_path, raw_path, gt_data=False, start_steps=0, end_steps=0):
        self.data_path = data_path
        self.gt_data = gt_data
        self.start_steps = start_steps
        self.end_steps = end_steps

        self.coords, self.matrices = self.load_matrix(start_steps, end_steps)
        self.clusters = self.load_cluster(start_steps, end_steps)

        self.num_coords = self.coords.shape[1]
        self.scale = self.get_scale() # for visualization, robot body graph and joints

        self.bounding_box = self.get_bounding_box(raw_path)

    def get_bounding_box(self, raw_path):
        print('raw_path', raw_path)
        # load all point clouds
        pc = o3d.geometry.PointCloud()

        # convert raw path to string
        raw_path = str(raw_path)
        paths = sorted(glob.glob(raw_path + '*/'))

        for path in paths:
            pcd = o3d.io.read_point_cloud(path + 'robot.ply')
            pc += pcd

        # get the bounding box
        bbox = pc.get_axis_aligned_bounding_box()

        #print("Bounding Box")
        #print(bbox.get_extent())

        # calculate diagonal length
        #print("Diagonal Length")
        dist = np.linalg.norm(bbox.get_extent())
        return dist

    def get_scale(self):
        """
        Get the scale of self.coords at time step 0
        """
        scale = []
        for i in range(3):
            scale.append(np.max(self.coords[0, :, i]) - np.min(self.coords[0, :, i]))
        scale = np.max(scale)
        return scale

    def load_matrix(self, start_steps=0, end_steps=0):
        # list .npy files in the data_path
        files = sorted(glob.glob(self.data_path + 'matrix/*.npy'))
        files = files[start_steps:end_steps]
        matrices = [np.load(file) for file in files]
        matrices = np.array(matrices)
        print(matrices.shape) # (time-step, num_points, 4, 4)
        # euler = False
        # if euler:
        #     # matrix to xyzrpy
        #     xyzrpy_t = []
        #     for i in range(matrices.shape[0]):
        #         xyzrpy_n = []
        #         for j in range(matrices.shape[1]):
        #             xyzrpy = matrix_to_xyzrpy_scipy(matrices[i, j])
        #             xyzrpy_n.append(xyzrpy)
        #         xyzrpy_t.append(xyzrpy_n)

        #     rot = np.array(xyzrpy_t)
        # else: # x,y,z,qx,qy,qz,qw
        xyzq_t = []
        for i in range(matrices.shape[0]):
            xyzq_n = []
            for j in range(matrices.shape[1]):
                # xyzq = matrix_to_xyzquat_scipy(matrices[i, j])

                # torch3d
                matrices_tensor = torch.tensor(matrices[i, j])
                q = matrix_to_quaternion(matrices_tensor[:3, :3]).numpy()
                xyz = matrices_tensor[:3, 3].numpy()
                xyzq = np.concatenate([xyz, q])
                xyzq_n.append(xyzq)
            xyzq_t.append(xyzq_n)
        rot = np.array(xyzq_t)

        return rot, matrices
    def load_cluster(self, start_steps=0, end_steps=0):
        # list .npz files in the data_path
        files = sorted(glob.glob(self.data_path + 'cluster/*.npz'))
        files = files[start_steps:end_steps]
        clusters = [np.load(file) for file in files] # list of list of np arrays, (time-step, num_coords, num_points, 3)
        # print(clusters[0]['0']) # key bug
        return clusters

    def coord_dist_map(self, diff=True):
        """
        num_seg x num_seg x time-step matrix, np array
        """
        # angle
        d_matrix_list = []
        time_steps = self.coords.shape[0]
        # these parameters can be tuned!
        LAMBDA_ROT = 1 / math.pi  # divide by maxmimum possible rotation angle (pi)
        # for LAMBDA_TRANS, assume that translation coeffs. are normalized in 3D eucl. space
        LAMBDA_TRANS = 1 / (1 * math.sqrt(3))  # divide by maximum possible translation (2 * unit cube diagonal)
        LAMBDA_BBOX = 1 / (self.bounding_box *2)

        print("LAMBDA_TRANS", LAMBDA_TRANS)
        print("LAMBDA_BBOX", LAMBDA_BBOX)
        d_matrix_xyz = np.zeros((self.num_coords, self.num_coords))
        d_matrix_rpy = np.zeros((self.num_coords, self.num_coords))
        trans_dist = np.zeros((self.num_coords, self.num_coords))
        rot_dist = np.zeros((self.num_coords, self.num_coords))
        if diff:
            trans_diff = np.diff(self.coords[:, :, :3], axis=0) # (time_step-1, num_coords, 3)
            # use rotvec
            rot_diff = np.zeros((time_steps-1, self.num_coords, 3))

            for i in range(time_steps-1):
                for j in range(self.num_coords):
                    # rot_diff[i, j] = signed_rotation_metric(self.matrices[i][j][:3, :3], self.matrices[i+1][j][:3, :3])
                    # geodesic  = roma.rotmat_geodesic_distance(torch.tensor(self.matrices[i][j][:3, :3]), torch.tensor(self.matrices[i+1][j][:3, :3]))
                    # print("signed_rotation_metric", rot_diff[i, j])
                    # print("geodesic", geodesic)
                    relative_rot = self.matrices[i][j][:3, :3].T @ self.matrices[i+1][j][:3, :3]
                    rot_diff[i, j] = roma.rotmat_to_rotvec(torch.tensor(relative_rot))
            
            for i in range(time_steps-1):
                for j in range(self.num_coords):
                    for k in range(self.num_coords):
                        d_matrix_xyz[j,k] = LAMBDA_BBOX * np.linalg.norm(trans_diff[i][j] - trans_diff[i][k], ord=2)
                        d_matrix_rpy[j,k] = LAMBDA_ROT * roma.utils.rotvec_geodesic_distance(torch.tensor(rot_diff[i][j]), torch.tensor(rot_diff[i][k]))
                        #print(d_matrix_xyz[j,k], d_matrix_rpy[j,k])
                # normalize then add
                #print(np.min(d_matrix_xyz), np.max(d_matrix_xyz))
                #print(np.min(d_matrix_rpy), np.max(d_matrix_rpy))

                for j in range(self.num_coords):
                    for k in range(self.num_coords):
                        trans_dist[j, k] = np.linalg.norm(d_matrix_xyz[j] - d_matrix_xyz[k], ord=2)
                        rot_dist[j, k] = np.linalg.norm(d_matrix_rpy[j] - d_matrix_rpy[k], ord=2)
                #trans_dist = (math.pi*trans_dist)/ np.max(trans_dist) if np.max(trans_dist) != 0 else trans_dist
                #rot_dist = rot_dist/ np.max(rot_dist) if np.max(rot_dist) != 0 else rot_dist

                d_matrix = trans_dist + rot_dist
                #print(d_matrix)
                d_matrix_list.append(d_matrix)
        else:
            # ori_trans_dist = np.zeros((self.num_coords, self.num_coords))
            # ori_rot_dist = np.zeros((self.num_coords, self.num_coords))
            # for j in range(self.num_coords):
            #     for k in range(self.num_coords):
            #         ori_trans_dist[j, k] = np.linalg.norm(self.coords[0][j][:3] - self.coords[0][k][:3], ord=2)
            #         ori_rot_dist[j, k] = roma.rotmat_geodesic_distance(torch.tensor(self.matrices[0][j][:3, :3]), torch.tensor(self.matrices[0][k][:3, :3]))

            for i in range(time_steps):
                for j in range(self.num_coords):
                    for k in range(self.num_coords):
                        d_matrix_xyz[j, k] = LAMBDA_BBOX * np.linalg.norm(self.coords[i][j][:3] - self.coords[i][k][:3], ord=2)
                        d_matrix_rpy[j, k] = LAMBDA_ROT * roma.rotmat_geodesic_distance(torch.tensor(self.matrices[i][j][:3, :3]), torch.tensor(self.matrices[i][k][:3, :3]))
                #d_matrix_xyz = (math.pi * d_matrix_xyz)/ np.max(d_matrix_xyz) if np.max(d_matrix_xyz) != 0 else d_matrix_xyz
                # d_matrix_rpy = d_matrix_rpy/ np.max(d_matrix_rpy) if np.max(d_matrix_rpy) != 0 else d_matrix_rpy
                # if i == 0:
                #     d_matrix = d_matrix_xyz + d_matrix_rpy - ori_rot_dist - ori_trans_dist
                # else:
                d_matrix = d_matrix_xyz + d_matrix_rpy
                d_matrix_list.append(d_matrix)

        coord_dist_map = np.stack(d_matrix_list, axis=2)
        sum_map = np.sum(np.abs(coord_dist_map), axis=2)

        return coord_dist_map, sum_map
    
    def coord_dist_map_legacy(self, diff=True):
        """
        40x40xtime-step matrix, np array
        """
        # angle
        d_matrix_list = []
        time_steps = self.coords.shape[0]

        for i in range(time_steps):
            # xyz - xyz at time step 0, rpy already = 0 at time step 0

            xyz_i = self.coords[i][:, :3] - self.coords[0][:, :3] # (num_coords, 3)
            # xyz_i = self.coords[i][:, :3] # works for solo8
            d_matrix_xyz = distance_matrix(xyz_i, xyz_i)
            d_matrix_rpy = distance_matrix(self.coords[i][:, 3:], self.coords[i][:, 3:])
            d_matrix = d_matrix_xyz + d_matrix_rpy
            d_matrix_list.append(d_matrix)

        coord_dist_map = np.stack(d_matrix_list, axis=2)
        sum_map = np.sum(np.abs(coord_dist_map), axis=2)
        # normalize the sum_map
        sum_map = (sum_map - np.min(sum_map)) / (np.max(sum_map) - np.min(sum_map))

        return coord_dist_map, sum_map
    
    def coord_mst(self):
        """
        Minimum Spanning Tree
        """
        sum_coord = np.sum(self.coords[:, :, :3], axis=0) # xyz
        dist_xyz_0 = distance_matrix(sum_coord, sum_coord) # (time_step, time_step)

        mst = nx.minimum_spanning_tree(nx.Graph(dist_xyz_0))
        edges_base = list(mst.edges)
        orignal_idx = np.arange(self.num_coords) 
        G_MST = nx.Graph()
        G_MST.add_nodes_from(orignal_idx)
        G_MST.add_edges_from(edges_base)

        return G_MST


    def kinematics_tree(self, g0, g1):
        """
        Build the kinematics tree based on the cluster_idx
        """
        links = []
        cluster_idx = list(nx.connected_components(g1))
        for link_id, idx in enumerate(cluster_idx):
            link = {
                'id': link_id,
                'tree_id': None,  # Initialize tree_id
                'cluster_idx': idx,
                'connected_links': set(), # empty set
            }

            for cid in idx:
                # find the direct connected clusters in base graph
                connected_cid = list(g0.neighbors(cid))
                # find the connected link id
                for i in range(len(cluster_idx)):
                    if i == link_id:
                        continue
                    for ccid in connected_cid:
                        if ccid in cluster_idx[i] and i not in link['connected_links']:
                            link['connected_links'].add(i)
            links.append(link)

        # we have a link graph at this point, need to check the graph is Acyclic (no cycle) and connected (all nodes are connected), to grantee the tree can be built given the root link
        link_graph = nx.Graph()
        link_graph.add_nodes_from(range(len(links)))
        for link in links:
            for connected_link in link['connected_links']:
                link_graph.add_edge(link['id'], connected_link)

        # check the graph is Acyclic and connected
        if nx.is_connected(link_graph) and nx.is_forest(link_graph):
            print("The graph is connected and Acyclic")
        else:
            print("The graph is not connected or Acyclic")
            # return

        # find the root link, the min xyz movement
        for link in links:
            centers = np.mean(self.coords[:, list(link['cluster_idx']), :], axis=1) # (time_step, 6)
            centers_diff = np.diff(centers, axis=0) # (time_step-1, 6)
            link['movement'] = np.sum(np.linalg.norm(centers_diff, axis=1))

        links = sorted(links, key=lambda x: x['movement'])
        for link in links:
            print(link)
        # build the tree
        root_link = links[0]
        root_link['parent_id'] = None
        root_link['tree_id'] = 0  # Root gets tree_id 0
        tree_id_counter = 1  # Initialize counter for tree_id
        # initialize
        current_layer = [root_link]
        count = 0
        while True:
            count += 1
            child_link_id = set()
            next_layer = []
            for current_link in current_layer:
                if current_link['parent_id'] is not None:
                    child = current_link['connected_links'] - {current_link['parent_id']}
                else:
                    child = current_link['connected_links']

                for i in child: # current link's child link id
                    # find the id = i link
                    for link in links:
                        if link['id'] == i:
                            link['parent_id'] = current_link['id']
                            link['tree_id'] = tree_id_counter  # Assign tree_id
                            tree_id_counter += 1  # Increment counter
                            next_layer.append(link)
                            break

                child_link_id.update(child)

            print(child_link_id)
            current_layer = next_layer

            if len(child_link_id) == 0 or count > 100: # break if no child link or exceed the max layer (dead loop)
                break

        # rank the links by tree_id
        links = sorted(links, key=lambda x: x['tree_id'])

        for i, link in enumerate(links):
            print(f'Layer{i} ---','Real ID: ', link['id'], 'Tree ID: ', link['tree_id'], 'Parent Link', link['parent_id'])
        return links
    
    def cluster_to_link(self, cluster_idx):
        """
        Combine the clusters to links
        links.coords: (time-step, num_links, 6) np array, xyzrpy
        links.clusters: (time-step, num_links, num_points, 3) np array
        """
        mesh_links = []
        time_steps = self.coords.shape[0]

        for link_id, idx in enumerate(cluster_idx):
            # initialize one link
            mesh_link = {
                'matrices': None, # (time-step, 4, 4)
                'clusters': [],  # (time-step, num_points, 3)
                'clusters_wf': []  # (time-step, num_points, 3)
            }
            # print(idx)
            link_filtered_coords = self.coords[:, list(idx)] # (time-step, num_coords_in_one_link, 6)
            link_coords = np.mean(link_filtered_coords, axis=1) # (time-step, 6)

            # link_filtered_matrices = self.matrices[:, list(idx)] # (time-step, num_coords_in_one_link, 4, 4)
            # print(link_filtered_matrices.shape)
            # euler = False
            # if euler == True:
            #     link_matrix = [xyzrpy_to_matrix_scipy(link_coords[i][:3], link_coords[i][3:]) for i in range(time_steps)] # (time-step, 4, 4)
            # else:
            link_matrix = [xyzquant2matrix_torch(link_coords[i]).numpy() for i in range(time_steps)]

            mesh_link['matrices'] = np.array(link_matrix)

            link_filtered_clusters = [] # (time-step, num_coords_in_one_link, num_points, 3)
            link_filtered_clusters_wf = [] # (time-step, num_coords_in_one_link, num_points, 3)
            for i in range(time_steps):
                # local frame
                filtered_clusters = {
                    str(key): self.clusters[i][str(key)] for key in idx
                }
                # world frame, combined
                filtered_clusters_wf = [
                    filtered_clusters[str(key)]@self.matrices[i][int(key)][:3, :3].T 
                    + self.matrices[i][int(key)][:3, 3] 
                    for key in idx
                ]
                link_cluster_wf = np.concatenate(filtered_clusters_wf, axis=0) # (num_points, 3)
                # print("link_cluster_wf", link_cluster_wf.shape)


                link_filtered_clusters.append(filtered_clusters)
                link_filtered_clusters_wf.append(filtered_clusters_wf)

                inv_matrix = np.linalg.inv(link_matrix[i]) 
                link_cluster_lf = link_cluster_wf@inv_matrix[:3, :3].T + inv_matrix[:3, 3] # (num_points, 3)
                # print("link_cluster_lf", link_cluster_lf.shape)

                mesh_link['clusters'].append(link_cluster_lf)
                mesh_link['clusters_wf'].append(link_cluster_wf)

            mesh_links.append(mesh_link) ## warning: two dicts are named link

        return mesh_links
    
    def visualize_cluster(self, edges, coord_idx=0, vis_params=None, save_params=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coords[coord_idx][:, :3])
        pcd.paint_uniform_color([0.1, 0.1, 0.1])
        # Create spheres
        spheres = []
        for i in range(self.num_coords):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([0.1, 0.1, 0.1])
            sphere.translate(self.coords[coord_idx][i][:3])
            spheres.append(sphere)
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self.coords[coord_idx][:, :3])
        line_set.lines = o3d.utility.Vector2iVector(edges)
        # Create robot graph
        robot_graph = lineset2cylinder(line_set, pcd, scale=self.scale)
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=844, height=773)
        # Add geometries
        for geom in robot_graph:
            vis.add_geometry(geom)
        # Set initial view
        ctr = vis.get_view_control()

        if vis_params is not None:
            ctr.set_zoom(vis_params[0])
            ctr.set_front(vis_params[1])
            ctr.set_lookat(vis_params[2])
            ctr.set_up(vis_params[3])
        # Run visualizer
        vis.run()

        # Capture final view parameters
        print(vis.get_view_status())
        vis.capture_screen_image(f"data/image/{ROBOT}_{int(time.time())}.png", do_render=True)
        # Close the window
        vis.destroy_window()

    def animate_cluster(self, cluster_idx, edges):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coords[0][:, :3])
        pcd.paint_uniform_color([0.1, 0.1, 0.1])
        # o3d.visualization.draw_geometries([pcd])
        num_links = len(cluster_idx)

        draw_graph = False
        if draw_graph:
            lines = []
            for edge in edges:
                lines.append([self.coords[0][edge[0]][:3], self.coords[0][edge[1]][:3]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(self.coords[0][:, :3])
            line_set.lines = o3d.utility.Vector2iVector(edges)

            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1600, height=1200)
            view_control = vis.get_view_control()
            view_control.set_zoom(1.2)
            view_control.set_lookat([0, 0, 0])
            view_control.set_front([0, -1, 1])
            view_control.set_up([0, 0, 1])
            vis.add_geometry(pcd)
            vis.add_geometry(line_set)
            for i in range(1, len(self.coords)):
                pcd.points = o3d.utility.Vector3dVector(self.coords[i][:, :3])
                lines = []
                for edge in edges:
                    lines.append([self.coords[i][edge[0]][:3], self.coords[i][edge[1]][:3]])
                line_set.points = o3d.utility.Vector3dVector(self.coords[i][:, :3])
                line_set.lines = o3d.utility.Vector2iVector(edges)
                vis.update_geometry(pcd)
                vis.update_geometry(line_set)
                vis.poll_events()
                vis.update_renderer()
                view_control.set_zoom(1.2)
                view_control.set_lookat([0, 0, 0])
                view_control.set_front([0, -1, 1])
                view_control.set_up([0, 0, 1])
                time.sleep(0.5)

        # visualize the point cloud
        # color_map = plt.get_cmap('jet')
        # colors = color_map(np.linspace(0, 1, len(cluster_idx)))[:,:3]
        colormap = plt.get_cmap("jet")
        colors = [colormap(i / (num_links)) for i in range(num_links)]
        # color sequence
        color_seq = np.zeros(self.num_coords) # color index sequence
        for i in range(len(cluster_idx)):
            cluster = cluster_idx[i]
            color_seq[list(cluster)] = i
        # time step 0
        cluster_pcds = self.clusters[0] # (num_coords, num_points, 3)
        pcds = []
        matrices = self.matrices[0]
        for i in range(self.num_coords):
            cluster_np = cluster_pcds[str(i)]
            print(cluster_np.shape)
            # local to world
            cluster_np = cluster_np@matrices[i][:3, :3].T + matrices[i][:3, 3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster_np)
            pcd.paint_uniform_color(colors[int(color_seq[i])][:3])

            # pcd.transform(matrices[i])
            pcds.append(pcd)
        o3d.visualization.draw_geometries(pcds)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=1200)
        view_control = vis.get_view_control()
        view_control.set_zoom(1.5)
        view_control.set_lookat([0, 0, 0])
        view_control.set_front([0, -1, 1])
        view_control.set_up([0, 0, 1])
        for pcd in pcds: vis.add_geometry(pcd)
        for i in range(1, len(self.coords)):
            cluster_pcds = self.clusters[0]
            matrices = self.matrices[i]
            for j in range(self.num_coords):
                cluster_np = cluster_pcds[str(j)] # yeah, it's strange
                pcds[j].points = o3d.utility.Vector3dVector(cluster_np)
                pcds[j].paint_uniform_color(colors[int(color_seq[j])][:3])

                pcds[j].transform(matrices[j])
            for pcd in pcds: 
                vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            view_control.set_zoom(1.5)
            view_control.set_lookat([0, 0, 0])
            view_control.set_front([0, 1, 0.5])
            view_control.set_up([0, 0, 1])
            time.sleep(0.5)

def main(vis_params):

    sub_part_path = sorted(glob.glob(PART_PATH + '*/'))[START_VIDEO:END_VIDEO]
    # sub_link_path = sorted(glob.glob(LINK_PATH + '*/'))
    sub_raw_path = RAW_PATH_LIST[START_VIDEO:END_VIDEO]
    if len(sub_part_path) == 0:
        # try real data
        sub_part_path = sorted(glob.glob("data/part/{ROBOT}_{NUM_SEG}_seg/*/"))
        sub_raw_path = sorted(glob.glob("data/raw/{ROBOT}_{NUM_SEG}_seg/*/"))
    print(sub_part_path)

    cm_list = []
    sum_map_list = []
    for i, path in enumerate(sub_part_path):

        cm = CoordMap(path, sub_raw_path[i], start_steps=START_STEPS, end_steps=END_STEPS)
        if LEGACY:
            _, sum_map = cm.coord_dist_map_legacy(diff=False)
        else:
            if DIFF:
                _, sum_map = cm.coord_dist_map(diff=True)
            else:
                _, sum_map = cm.coord_dist_map(diff=False)
        cm_list.append(cm)
        sum_map_list.append(sum_map)

    sum_map = np.mean(sum_map_list, axis=0)
    #print("max:"    , max(sum_map.flatten()))

    # normalize the sum_map
    sum_map = (sum_map - np.min(sum_map)) / (np.max(sum_map) - np.min(sum_map))

    plt.figure()
    plt.imshow(1-sum_map, cmap='Blues')
    plt.colorbar()
    plt.show()

    g0 = cm_list[0].coord_mst()

    if UNKNOWN_DOF:
        # test_num_links = (3, int(cm_list[0].num_coords/2))
        test_num_links = (4, min(25, cm_list[0].num_coords)) 
        # from 3 to 20 links, include all cases in current dataset
        cluster_idx, g1, s_score_list, nls = silhouette_score_method(cm_list[0].num_coords, sum_map, link_range=test_num_links)
        if len(sub_part_path) == 1:
            print(sub_part_path[0] + 'score/')
            os.makedirs(sub_part_path[0] + 'score/', exist_ok=True)
            
            with open(sub_part_path[0] + 'score/silhouette_score.txt', 'w') as f:
                f.write(f"Silhouette Score: {s_score_list}\n")
                f.write(f"Number of Links: {nls}\n")

        plt.figure()
        plt.plot(nls, s_score_list)
        plt.xlabel('Number of Links')
        plt.ylabel('Silhouette Score')
        #plt.show()

        plt.grid(True)  # Add grid for better readability
        plt.savefig(sub_part_path[0] + 'score/silhouette_score_plot.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        dof = len(cluster_idx) - 1 # number of links - 1
        
    else:
        dof = DOF
        cluster_idx, g1, s_score = coord_clustering(cm_list[0].num_coords, sum_map, num_links=dof+1)

    edges_base = list(g0.edges)
    edges = list(g1.edges)
    cm_list[0].visualize_cluster(edges_base, coord_idx=0, vis_params=vis_params)
    cm_list[0].visualize_cluster(edges, coord_idx=0, vis_params=vis_params)
    cm_list[0].animate_cluster(cluster_idx, edges)

    links = cm_list[0].kinematics_tree(g0, g1)

    """TODO: save tree here"""

    joint_data = estimate_joint_axes_from_tree(links, cm_list, START_STEPS, END_STEPS-START_STEPS, 4)

    # link to mesh
    path_list = [path.split('/')[-2] for path in sub_part_path]
    sub_link_path = [LINK_PATH + path + '/' for path in path_list][:1] # only one video for meshing process
    save_links(cm_list, cluster_idx, sub_link_path, START_STEPS, END_STEPS) # save clusters and matrices
    refine_links_clusters(sub_link_path, START_STEPS, END_STEPS, dof) # refine clusters icp
    visualize_links(sub_link_path, START_STEPS, END_STEPS, dof, VIS_FLOW) # load clusters and matrices, visualize in world frame
    link_mesh(sub_link_path, dof, VSIZE, VIS_FLOW)  # load clusters, convert to voxel grids, convert to mesh, save mesh

    visualize_kinematic_tree(cm_list[0], links, joint_data, scale=cm_list[0].scale, vis_params=vis_params, robot=ROBOT)
    os.makedirs(f'data/urdf/{ROBOT}_{NUM_SEG}_seg/', exist_ok=True)
    urdf_path = f'data/urdf/{ROBOT}_{NUM_SEG}_seg/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams.urdf'
    create_urdf(links, joint_data,cm_list[0], urdf_path, sub_link_path[0]) 
    visualize_urdf(urdf_path, ori=ORI)

if __name__ == "__main__":
# parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='wx200_5')
    parser.add_argument('--xyz_r', type=float, default=0.5)
    parser.add_argument('--start_steps', type=int, default=0)
    parser.add_argument('--end_steps', type=int, default=10)  # number of frames/steps
    parser.add_argument('--start_video', type=int, default=0)
    parser.add_argument('--end_video', type=int, default=1)  # number of videos, each video is a sequence of frames
    parser.add_argument('--unknown_dof', action='store_true')
    parser.add_argument('--vis_flow', action='store_true')
    parser.add_argument('--num_cameras', type=int, default=20) # number of cameras
    parser.add_argument('--step_size', type=int, default=4) # motor step size
    parser.add_argument('--diff', action='store_true')
    parser.add_argument('--legacy', action='store_true')
    args = parser.parse_args()
    # get json parameters by robot name
    with open('parameters.json') as f:
        params = json.load(f)
    robot_params = params[args.robot]
    vis_params = []
    vis_params.append(robot_params['zoom'])
    vis_params.append(robot_params['front'])
    vis_params.append(robot_params['lookat'])
    vis_params.append(robot_params['up'])
    print(vis_params)

    # json parameters
    ROBOT = args.robot
    NUM_SEG = robot_params['num_seg']
    ORI = robot_params['ori']
    VSIZE = robot_params['voxel_size']
    DIFF = args.diff
    LEGACY = args.legacy

    VIS_FLOW = args.vis_flow
    STEP_SZIE = args.step_size
    NUM_CAMERAS = args.num_cameras
    START_STEPS = args.start_steps
    END_STEPS = args.end_steps
    START_VIDEO = args.start_video
    END_VIDEO = args.end_video
    UNKNOWN_DOF = args.unknown_dof

    if not UNKNOWN_DOF:
        DOF = robot_params['dof']
    
    PART_PATH = f'data/part/{ROBOT}_{NUM_SEG}_seg/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/'
    LINK_PATH = f'data/mesh/{ROBOT}_{NUM_SEG}_seg/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/'
    RAW_PATH_LIST = sorted(glob.glob(f'data/raw/{ROBOT}/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/*/'))
    if len(RAW_PATH_LIST) == 0: # for real data
        RAW_PATH_LIST = sorted(glob.glob(f'data/raw/{ROBOT}/*/'))
    print("RAW_PATH_LIST", RAW_PATH_LIST)

    main(vis_params)
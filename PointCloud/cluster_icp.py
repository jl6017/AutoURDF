import open3d as o3d
import numpy as np
import sklearn.cluster as cluster
from glob import glob
from scipy.spatial.transform import Rotation as R

def xyzrpy_to_matrix_scipy(xyz, rpy):
    rotator = R.from_euler('xyz', rpy)
    transformation = np.eye(4)
    transformation[:3, 3] = xyz
    transformation[:3, :3] = rotator.as_matrix()
    return transformation

class Segments:
    """
    segments of point cloud
    1. load point cloud, a list of .ply files
    2. k-means clustering for the starting config, n sub pc, n center frames
    3. in one iteration, 
        1. for each sub pc, calculate icp transformation
        2. find a nearest neighbor, if the center frame transfer change is less than a threshold, merge
    4. visualize the segments

    """
    def __init__(self, data_path, sample_size=None) -> None:
        self.pc_path = data_path
        self.pc_list = []
        self.init_coord_list = []
        self.init_matrix_list = []
        self.init_segment_list = []
        self.data_size = 0
        self.sample_size = sample_size
        self._load_pc()

    def _load_pc(self):
        # list all sub dir
        sub_path = sorted(glob(self.pc_path + '*/'))
        self.data_size = len(sub_path)
        print(f'Found {self.data_size} sub path')
        for path in sub_path:
            pc = o3d.io.read_point_cloud(path + 'robot.ply')
            if self.sample_size is not None:
                pc = pc.farthest_point_down_sample(self.sample_size) # down sample if given sample_size
            self.pc_list.append(pc)


    def k_means_cluster(self, pc_id=0, num=30, normal= False, colors=None):
        pc = self.pc_list[pc_id]
        if normal:
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pc.orient_normals_consistent_tangent_plane(30)
            # pc = pc.farthest_point_down_sample(5000)

            # visualize pc with normals
            o3d.visualization.draw_geometries([pc], point_show_normal=True)

            pc_np = np.asarray(pc.points)
            # add normal
            normals = np.array(pc.normals) * 0.5
            print("n", normals.shape)
            pc_normal_np = np.hstack([pc_np, normals])
            print(pc_normal_np.shape)
            sub_pcd = cluster.k_means(pc_normal_np, init="k-means++", n_clusters=num)
        # cluster_func = cluster.AgglomerativeClustering(n_clusters=num, linkage='complete')
        else:
            pc_np = np.asarray(pc.points)
            sub_pcd = cluster.k_means(pc_np, init="k-means++", n_clusters=num)

        
        new_pcd = []
        coords = []

        for i in range(num):
            if colors is not None:
                i_color = colors[i]
            else:
                i_color = np.random.rand(3)
            mask = sub_pcd[1] == i
            # mask = sub_pcd.labels_ == i
            i_pcd_np = pc_np[mask]
            i_pcd = o3d.geometry.PointCloud()
            i_pcd.points = o3d.utility.Vector3dVector(i_pcd_np)
            i_pcd.paint_uniform_color(i_color)
            new_pcd.append(i_pcd)

            center = np.mean(i_pcd_np, axis=0)

            # init coord frame, x, y, z, r, p, y
            # rpy = (np.random.rand(3) - 0.5) * 0.1 * np.pi
            # xyzrpy = np.array([center[0], center[1], center[2], rpy[0], rpy[1], rpy[2]])
            rpy = np.zeros(3)
            xyzrpy = np.array([center[0], center[1], center[2], rpy[0], rpy[1], rpy[2]])
            self.init_coord_list.append(xyzrpy)
            matrix = xyzrpy_to_matrix_scipy(xyzrpy[:3], xyzrpy[3:])
            self.init_matrix_list.append(matrix)
            inv_matrix = np.linalg.inv(matrix) 
            init_segment = inv_matrix @ np.hstack([i_pcd_np, np.ones((i_pcd_np.shape[0], 1))]).T
            init_segment = init_segment[:3].T
            self.init_segment_list.append(init_segment) # local frame

            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            coord.transform(matrix)
            coords.append(coord)


        o3d.visualization.draw_geometries(coords + new_pcd)
        return new_pcd

    def visualize(self, pcs):
        colors = np.linspace(0.5, 0, len(pcs))
        color_pcs = []
        for i, pc in enumerate(pcs):
            pc.paint_uniform_color([colors[i], colors[i], colors[i]])
            color_pcs.append(pc)
        o3d.visualization.draw_geometries(color_pcs)


def masked_icp(clusters_local, clusters_world, step_pc_np, matrices, visual=False, ori=False, scale=1.2, th=1, colors=None):
    """
    Args:
    - clusters_local: list of np array, (N, 3), segments(clusters) in local coords
    - clusters_world: list of np array, (N, 3), segments(clusters) in world coords
    - step_pc_np: np array, (N, 3), robot point cloud in world coords
    - matrices: list of np array, (4, 4), transformation matrix from local to world
    - visual: visualization flag
    - ori: only update the rotation part of the matrix
    """
    world_clusters = []
    new_matrices = []
    color_i = 0
    for c_loacl, c_world, matrix in zip(clusters_local, clusters_world, matrices):
        # box, xmin, xmax, ymin, ymax, zmin, zmax
        box = np.array([[np.min(c_world[:, 0]), np.max(c_world[:, 0])], 
                        [np.min(c_world[:, 1]), np.max(c_world[:, 1])],
                        [np.min(c_world[:, 2]), np.max(c_world[:, 2])]])
        
        # scale the box
        box_center = np.mean(box, axis=1) # (3,)
        box_size = box[:, 1] - box[:, 0] # (3,)
        box = np.vstack([box_center - 0.5 * scale * box_size, box_center + 0.5 * scale * box_size]).T # (3, 2)
        
        mask = np.logical_and(step_pc_np[:, 0] > box[0,0], step_pc_np[:, 0] < box[0,1])
        mask = np.logical_and(mask, step_pc_np[:, 1] > box[1,0])
        mask = np.logical_and(mask, step_pc_np[:, 1] < box[1,1])
        mask = np.logical_and(mask, step_pc_np[:, 2] > box[2,0])
        mask = np.logical_and(mask, step_pc_np[:, 2] < box[2,1])
        
        masked_pc = step_pc_np[mask]
        masked_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(masked_pc))
        masked_pc.paint_uniform_color([1, 0, 0])
        c_local_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c_loacl))
        c_local_pc.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([masked_pc, segment_pc])

        # ICP
        threshold = th
        reg_p2p = o3d.pipelines.registration.registration_icp(c_local_pc, masked_pc, threshold, matrix, 
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
        
        if ori:
            icp_matrix = reg_p2p.transformation.copy()
            icp_matrix[:3, 3] = matrix[:3, 3]
        else:
            icp_matrix = reg_p2p.transformation.copy()

        c_local_pc.transform(icp_matrix)
        if colors is not None:
            c_local_pc.paint_uniform_color(colors[color_i])
            color_i += 1
        else:
            c_local_pc.paint_uniform_color(np.random.rand(3))
        world_clusters.append(c_local_pc)  # world coords

        # update matrix
        new_matrices.append(icp_matrix) # world to local

        # o3d.visualization.draw_geometries([masked_pc, segment_pc])

    # step_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(step_pc_np))
    # step_pc.paint_uniform_color([0, 0, 1])

    if visual:
        o3d.visualization.draw_geometries(world_clusters) # + [step_pc]

    # to numpy
    w_clusters_np = [np.asarray(cluster.points) for cluster in world_clusters]

    new_matrices = np.array(new_matrices)

    return w_clusters_np, new_matrices


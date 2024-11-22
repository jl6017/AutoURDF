import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from coord_map import CoordMap
import transforms3d
from numpy.linalg import svd
from scipy import optimize
from visualize import visualize_kinematic_tree_with_axes
from glob import glob

def get_cluster_pose_mean(cm, cluster, step):
    cluster_coords = cm.coords[step, cluster, :]
    
    # Calculate average position
    avg_pos = np.mean(cluster_coords[:, :3], axis=0)
    
    # Handle orientations
    rotations = R.from_euler('xyz', cluster_coords[:, 3:])
    
    # Use quaternion averaging for better interpolation
    quats = rotations.as_quat()
    avg_quat = average_quaternions(quats)
    
    # Convert average quaternion back to Euler angles
    avg_rotation = R.from_quat(avg_quat)
    avg_ori = avg_rotation.as_euler('xyz')
    
    return avg_pos, avg_ori

def average_quaternions(quaternions):
    """
    Calculate the average quaternion using the method described in
    https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
    """
    n = quaternions.shape[0]
    q = quaternions.T
    A = np.zeros((4, 4))
    
    for i in range(n):
        A += np.outer(q[:, i], q[:, i].T)
    
    A /= n
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Return the eigenvector corresponding to the largest eigenvalue
    return eigenvectors[:, -1]

def identify_base_cluster(cluster_idx, start_step, num_steps):
    clusters = [list(cluster) for cluster in cluster_idx]
    total_motion = []

    for cluster in clusters:
        motion = 0
        for step in range(start_step, start_step + num_steps):
            for cm in cms:
                pos, _ = get_cluster_pose_mean(cm, cluster, step)
                if step > start_step:
                    motion += np.linalg.norm(pos - prev_pos)
                prev_pos = pos
        total_motion.append(motion)

    return np.argmin(total_motion) # Cluster with the least motion is the base cluster

def relative_transform(pose_parent, pose_child):
        pos_parent, ori_parent = pose_parent
        pos_child, ori_child = pose_child
        R_parent = R.from_euler('xyz', ori_parent).as_matrix()
        R_child = R.from_euler('xyz', ori_child).as_matrix()
        T_parent = np.eye(4)
        T_parent[:3, :3] = R_parent
        T_parent[:3, 3] = pos_parent
        T_child = np.eye(4)
        T_child[:3, :3] = R_child
        T_child[:3, 3] = pos_child
        return np.linalg.inv(T_parent) @ T_child

def calculate_joint_axis_relative(poses_parent, poses_child):
    """
    Calculate the estimated joint axis between two links

    Args:
    - poses_parent: the poses of the parent link over all time steps, (position, orientation)
    - poses_child: the poses of the child link over all time steps, (position, orientation)

    Returns:
    - axes: the estimated joint axis over all time steps
    - angles: the estimated joint angle over all time steps
    - poses: the estimated joint position over all time steps

    """
    def init_position(initial_pos, joint_axes):
            """
            Refine the joint position by minimizing the distance to both parent and child link
            """
            # find largest abs component of initial_pos
            max_idx = np.argmax(np.abs(joint_axes))

            n = initial_pos[max_idx]/joint_axes[max_idx]

            return initial_pos - n * joint_axes
    axes = []
    angles = []
    poses = []

    for i in range(1, len(poses_parent)):
        #print(f"Step: {i}")
        """ # Calculate relative transformations in parent's frame at each time step
        T_relative_prev = relative_transform(poses_parent[i-1], poses_child[i-1])
        T_relative_curr = relative_transform(poses_parent[i], poses_child[i])
        
        # affine matrix to rotations encoded by axis vector and angle scalar

        T_relative = np.linalg.inv(T_relative_prev) @ T_relative_curr """

        # Calculate relative transformations in parent's frame at each time step
        T_r = relative_transform(poses_parent[i-1], poses_parent[i])
        #print(f"Relative transform: {T_r}")
        T_relative_childi_1 = relative_transform(poses_parent[i-1], poses_child[i-1])
        T_relative_childi = relative_transform(poses_parent[i-1], poses_child[i])
        
        # Calculate relative transformation between child links
        T_r2 = np.linalg.inv(T_r) @ T_relative_childi

        # Calculate relative transformation between child links at previous time step
        T_r1 = np.linalg.inv(T_relative_childi_1) @ T_r2
        #print(f"Relative transform: {T_r1}")


        # Extract rotation axis and angle and point
        axis, angle, pos = transforms3d.axangles.aff2axangle(T_r1)

        #axis, angle, pos = transforms3d.axangles.aff2axangle(T_relative)

        # print(f"Axis: {axis}")
        # print(f"Angle: {angle}")
        # print(f"Pos: {pos[:3]}")
        axes.append(axis)
        angles.append(angle)
        initial_pos = init_position(pos[:3], axis)
        pos_opt = initial_pos
        #pos_opt = refine_position(initial_pos, poses_parent[i-1][0], poses_child[i-1][0], axis)
        #print(f"Optimized pos: {pos_opt}")
        poses.append(pos_opt)

    return axes, angles, poses

def optimize_joint_axis(poses_parent, poses_child, axes, poses):
    """
    Calculate the principal axis of rotation for a joint between two links by averaging all axes

    Args:
    - poses_parent: the poses of the parent link over all time steps, (position, orientation)
    - poses_child: the poses of the child link over all time steps, (position, orientation)

    Returns:
    - principal_axis: the principal axis of rotation for the joint in the parent frame, (x, y, z)
    - global_axes: the principal axis of rotation in the global frame for visualization, (x, y, z)
    """
    
    def xyz_rpy_to_matrix(coord):
        xyz, rpy = coord
        rot = R.from_euler('xyz', rpy)
        rot_matrix = rot.as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = xyz
        return transform
    
    def error_function(joint_pos_parent, axes):
        """
        Calculate the deviation of the distances between the joint position and the child link over all time steps
        Args:
        - joint_pos_parent: the joint position in the parent frame, (x, y, z)
        Returns:
        - deviation: the deviation of the distances over all time steps
        """
        distances = []
        #print(f"axes: {axes}")
        for pose_parent, pose_child in zip(poses_parent, poses_child):
            pos_parent, ori_parent = pose_parent
            pos_child, ori_child = pose_child
            # parent to child vector in parent frame
            child_in_parent = np.array(pos_child - pos_parent)
            child_in_parent = np.linalg.inv(R.from_euler('xyz', ori_parent).as_matrix()) @ child_in_parent
            # Calculate vector from joint position to child in parent frame
            vector = child_in_parent - joint_pos_parent
            # Calculate the distance
            distance = np.linalg.norm(vector)
            distances.append(distance)
        # Calculate the deviation of distances over all time steps
        #print(f"Distances: {distances}")
        mean_distance = np.mean(distances)
        deviation = np.sum((np.array(distances) - mean_distance)**2)/len(distances)

        # add joint axes deviation to prevent perpendicular axes that make no distance change
        mean_axis = np.mean(axes, axis=0)
        deviation += np.sum((np.array(axes) - mean_axis)**2)/len(axes)
        return deviation
    
    def refine_position(initial_pos, parent_pos, child_pos, joint_axis):
        """
        Refine the joint position by finding the point along the joint axis
        that minimizes the sum of squared distances to both parent and child links.
        """
        def distance_sum(t):
            # Calculate position along the axis
            pos = initial_pos + t * joint_axis
            # Calculate sum of squared distances to parent and child
            dist_parent = np.linalg.norm(parent_pos - pos)
            dist_child = np.linalg.norm(child_pos - pos)

            return dist_parent + dist_child

        # Use scipy's minimize to find the optimal position along the axis
        result = optimize.minimize_scalar(distance_sum)
        
        # Calculate the optimized position
        optimized_pos = initial_pos + result.x * joint_axis
        
        return optimized_pos, result.fun

    # Normalize direction of axes to be consistent， and use svd to find the principal axis
    processed_axes = []
    for axis in axes:
        if np.abs(np.arccos(np.dot(axis, axes[0]))) > np.pi/2:
            axis = -axis
            #print("Invert axis")
        #print(f"Axis: {axis}")
        processed_axes.append(axis)

    deviation = error_function(poses, processed_axes)
    # Stack processed axes into a matrix
    axes_matrix = np.stack(processed_axes)
    print(f"length of axes: {len(axes)}")
    # Perform SVD
    U, S, Vt = svd(axes_matrix.T)

    # The principal axis is the first column of U
    principal_axis = U[:, 0]

    # Ensure the principal axis points in the same general direction as the original axes
    if np.dot(principal_axis, processed_axes[0]) < 0:
        principal_axis = -principal_axis

    print(f"Principal axis: {principal_axis}")

    principal_pos = np.mean(poses, axis=0)

    # Calculate global axes for visualization
    child_orientations = [pose[1] for pose in poses_child]
    child_rotations = [R.from_euler('xyz', ori).as_matrix() for ori in child_orientations]
    global_axes = [rotation @ principal_axis for rotation in child_rotations]

    # transform the homogeneous position coordinates to the global frame
    T_childs = [xyz_rpy_to_matrix(poses) for poses in poses_child]
    #print(T_childs)

    # convert to homogeneous coordinates
    principal_pos = np.concatenate([principal_pos, [1]])
    print(f"Principal pos: {principal_pos}")

    global_poses = [T_child @ principal_pos for T_child in T_childs]

    # from homogeneous to xyz
    global_poses = [pose[:3] for pose in global_poses]

    global_pos, distance = refine_position(global_poses[0], poses_parent[0][0], poses_child[0][0], principal_axis)

    # transform the global position back
    principal_pos = np.concatenate([global_pos, [1]])
    principal_pos = np.linalg.inv(T_childs[0]) @ principal_pos

    # average child to parent position
    parent_position = np.mean([pose[0] for pose in poses_parent], axis=0)
    child_position = np.mean([pose[0] for pose in poses_child], axis=0)
    principal_pos = principal_pos[:3]
    principal_pos += child_position - parent_position
    print(f"Refined principal pos: {principal_pos}")
    print(f"Distance: {distance}")

    return principal_axis, global_axes, global_pos,  principal_pos, deviation

def estimate_joint_axes(cluster_idx, start_step=0, num_steps=500, interval=1):
    """
    Estimate the joint axes between pairs of clusters
    Args:
    - cluster_idx: the indices of the clusters
    - start_step: the starting time step
    - num_steps: the number of time steps to consider
    - interval: the number of steps to skip between each data point
    Returns:
    - joint_data: a list of dictionaries containing the joint data, including the clusters, global position, global axis, and consistency
    """
    clusters = [list(cluster) for cluster in cluster_idx]
    n_clusters = len(clusters)
    joint_data = []
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            all_poses_i = []
            all_poses_j = []
            all_axes = []
            all_poses = []
            for cm in cms:
                for a in range(interval):
                    poses_i = []
                    poses_j = []
                    for step in range(start_step + a, start_step + num_steps, interval):
                        #print(f"Step: {step}")
                        pos_i, ori_i = get_cluster_pose_mean(cm, clusters[i], step)
                        pos_j, ori_j = get_cluster_pose_mean(cm, clusters[j], step)
                        poses_i.append((pos_i, ori_i))
                        poses_j.append((pos_j, ori_j))
                    axes, angles, poses = calculate_joint_axis_relative(poses_i, poses_j)
                    #print(f"Poses: {poses}")
                    all_poses_i.extend(poses_i)
                    all_poses_j.extend(poses_j)
                    all_axes.extend(axes)
                    all_poses.extend(poses)

            print(f"Joint between clusters {i} and {j}:")
            
            local_axis, global_axes, global_pos, local_pose, deviation = optimize_joint_axis(all_poses_i, all_poses_j, all_axes, all_poses)
            #_, _ = calculate_joint_axis(poses_i, poses_j)
            #joint_pos_local, deviation, joint_pos_global = optimize_joint_position(poses_i, poses_j, local_axis, local_pose[0][:3])
            
            print(f" dev: {deviation:.4f}")

            joint_data.append({
                'clusters': (i, j),
                'local_axis': local_axis,
                'local_pos': local_pose,
                'global_pos': global_pos,
                'global_axis': global_axes[0], # first time step orientation for visualization
                'deviation': deviation,
            })

    return joint_data

def construct_kinematic_tree_multi(joint_data, base_cluster, deviation_threshold=0.04):
    """
    Construct a kinematic tree from the joint data, allowing multiple children only for the base link
    Args:
    - joint_data: a list of dictionaries containing the joint data
    - base_cluster: the index of the base cluster
    - deviation_threshold: threshold for considering a connection as a direct child (only for base link)
    Returns:
    - tree: a dictionary representing the kinematic tree
    """
    # Create a graph representation
    graph = defaultdict(list)
    for joint in joint_data:
        i, j = joint['clusters']
        graph[i].append((j, joint['deviation'], joint['local_axis'], joint['local_pos']))
        graph[j].append((i, joint['deviation'], joint['local_axis'], joint['local_pos']))

    # Initialize the tree with the base cluster
    tree = {base_cluster: []}
    visited = set([base_cluster])
    current_layer = [base_cluster]

    while current_layer:
        next_layer = []
        for node in current_layer:
            potential_children = []
            for neighbor, deviation, axis, pos in graph[node]:
                if neighbor not in visited:
                    potential_children.append({'child': neighbor, 'deviation': deviation, 'axis': axis, 'pos': pos})

            # Sort potential children by deviation (lowest first)
            potential_children.sort(key=lambda x: x['deviation'])

            if node == base_cluster:
                # For base link, add all children below the threshold
                for child in potential_children:
                    if child['deviation'] < deviation_threshold:
                        tree[node].append(child)
                        visited.add(child['child'])
                        tree[child['child']] = []
                        next_layer.append(child['child'])
            elif potential_children:
                # For other nodes, add only the best child, regardless of threshold
                best_child = potential_children[0]
                tree[node].append(best_child)
                visited.add(best_child['child'])
                tree[best_child['child']] = []
                next_layer.append(best_child['child'])

        current_layer = next_layer

    return tree

def print_kinematic_tree(tree, node=None, depth=0):
    if node is None:
        node = list(tree.keys())[0]  # Start with the root
    
    print('  ' * depth + f"Cluster {node}")
    for child in tree[node]:
        print('  ' * (depth + 1) + f"└─ Child {child['child']} (deviation: {child['deviation']:.4f})")
        print('  ' * (depth + 1) + f"   - Axis: {child['axis']}")
        print('  ' * (depth + 1) + f"   - Position: {child['pos']}")
        print_kinematic_tree(tree, child['child'], depth + 2)

def analyze_kinematics(cluster_idx, start_step=0, num_steps=50, interval=1):

    base_cluster = identify_base_cluster(cluster_idx, start_step, num_steps)
    joint_data = estimate_joint_axes(cluster_idx, start_step, num_steps, interval)

    kinematic_tree = construct_kinematic_tree_multi(joint_data, base_cluster)

    print("\nKinematic tree:")
    print_kinematic_tree(kinematic_tree)
    return base_cluster, joint_data, kinematic_tree

if __name__ == "__main__":
    gt_data = False

    if gt_data:
        start_steps = 0
        end_steps = 50
        data_path = 'data/03012_init'
        cm = CoordMap(data_path, gt_data=True)
        cluster_idx = cm.generate_fake_cluster_idx()
        cms = [cm]

    else:
        # data_path = 'data/part/allegro_1d_30/' # 'PointCloud/temp(0)_wx200/'
        data_path = 'data/part/wx200_real_surf_15(False)1/'
        data_paths = sorted(glob(data_path + '*/')) # 'data_20seg_400p_*s/'
        data_paths = data_paths[:]
        start_steps = 0
        end_steps = 100
        cms = []
        sum_maps = []

        for data_path in data_paths:
            print(data_path)
            cm = CoordMap(data_path, start_steps=start_steps, end_steps=end_steps)
            cms.append(cm)
            _, _, sum_map, _ = cm.coord_dist_map(diff=False)
            sum_maps.append(sum_map)

        cm = cms[0]

        # Average the sum maps
        sum_map = sum(sum_maps) / len(sum_maps)
        print(sum_map)

        cluster_idx, edges, _ = cm.coord_cluster(sum_map, num_links=5) # TODO: function deleted

    # Usage
    base_cluster, joint_data, kinematic_tree = analyze_kinematics(cluster_idx, start_step=start_steps, num_steps=(end_steps-start_steps), interval=3)

    # Print results
    print(f"Base cluster: {base_cluster}")

    # Visualize kinematic tree
    visualize_kinematic_tree_with_axes(cm, cluster_idx, kinematic_tree, joint_data, time_step=0)
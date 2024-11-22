import numpy as np
import transforms3d
from scipy import optimize
import matplotlib.pyplot as plt
from helper_functions import matrix2xyzquant_torch, xyzquant2matrix_torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from scipy.spatial.transform import Rotation as R
import torch

def get_cluster_pose_mean(cm, cluster, step):
    cluster_coords = cm.coords[step, cluster, :]
    
    # Calculate average position
    avg_pos = np.mean(cluster_coords[:, :3], axis=0)
    
    # Handle orientations
    avg_quat = average_quaternions(cluster_coords[:, 3:])
    
    return avg_pos, avg_quat

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

def relative_transform(pose_parent, pose_child):
        pos_parent, ori_parent = pose_parent
        pos_child, ori_child = pose_child
        R_parent = quaternion_to_matrix(torch.tensor(ori_parent)).numpy()
        R_child =  quaternion_to_matrix(torch.tensor(ori_child)).numpy()
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
    def normalize_and_find_principal_axis(axes):
        # Normalize direction of axes to be consistent
        processed_axes = []
        reference_axis = axes[0] / np.linalg.norm(axes[0])
        
        for axis in axes:
            normalized_axis = axis / np.linalg.norm(axis)
            # Use dot product directly instead of arccos
            if np.dot(normalized_axis, reference_axis) < 0:
                normalized_axis = -normalized_axis
            processed_axes.append(normalized_axis)
        
        # Convert list to numpy array
        processed_axes = np.array(processed_axes)
        
        # Use SVD to find the principal axis
        U, _, _ = np.linalg.svd(processed_axes.T)
        principal_axis = U[:, 0]
        if np.dot(principal_axis, processed_axes[0]) < 0:
            principal_axis = -principal_axis
        return principal_axis

    # Calculate the principal axis of rotation
    principal_axis = normalize_and_find_principal_axis(axes)

    print(f"Principal axis: {principal_axis}")

    principal_pos = np.mean(poses, axis=0)

    # Calculate global axes for visualization
    child_orientations = [pose[1] for pose in poses_child]
    child_rotations = [quaternion_to_matrix(torch.tensor(ori)).numpy() for ori in child_orientations]
    global_axes = [rotation @ principal_axis for rotation in child_rotations]

    # transform the homogeneous position coordinates to the global frame
    T_childs = [xyzquant2matrix_torch(torch.cat([torch.tensor(poses[0]), torch.tensor(poses[1])])).numpy() for poses in poses_child]
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

    global_pos = T_childs[0] @ principal_pos
    global_pos = global_pos[:3]

    return principal_axis, global_axes, global_pos,  principal_pos

def estimate_joint_axes_from_tree(links, cm_list, start_step=0, num_steps=500, interval=1):
    """
    Estimate the joint axes between connected links in the kinematics tree
    Args:
    - links: the list of links from the kinematics tree
    - start_step: the starting time step
    - num_steps: the number of time steps to consider
    - interval: the number of steps to skip between each data point
    Returns:
    - joint_data: a list of dictionaries containing the joint data, including the clusters, global position, global axis, and consistency
    """
    joint_data = []
    for link in links:
        if link['parent_id'] is not None:  # Skip the root link
            parent_link = next(l for l in links if l['id'] == link['parent_id'])
            print(f"Parent link: {parent_link['id']}")
            all_poses_parent = []
            all_poses_child = []
            all_axes = []
            all_poses = []
            
            for cm in cm_list:
                for a in range(interval):
                    poses_parent = []
                    poses_child = []
                    for step in range(start_step + a, start_step + num_steps, interval):
                        #visualize_cluster_poses(cm, list(parent_link['cluster_idx']), step)
                        pos_parent, ori_parent = get_cluster_pose_mean(cm, list(parent_link['cluster_idx']), step)
                        pos_child, ori_child = get_cluster_pose_mean(cm, list(link['cluster_idx']), step)

                        poses_parent.append((pos_parent, ori_parent))
                        poses_child.append((pos_child, ori_child))
                    
                    axes, angles, poses = calculate_joint_axis_relative(poses_parent, poses_child)
                    all_poses_parent.extend(poses_parent)
                    all_poses_child.extend(poses_child)
                    all_axes.extend(axes)
                    all_poses.extend(poses)
            
            print(f"Joint between parent link {parent_link['id']} and child link {link['id']}:")
            
            local_axis, global_axes, global_pos, local_pose = optimize_joint_axis(all_poses_parent, all_poses_child, all_axes, all_poses)
            
            joint_data.append({
                'parent_link': parent_link['id'],
                'child_link': link['id'],
                'local_axis': local_axis,
                'local_pos': local_pose,
                'global_pos': global_pos,
                'global_axis': global_axes[0],  # first time step orientation for visualization
            })
    
    return joint_data

import xml.etree.ElementTree as ET
import os
import open3d as o3d

def create_urdf(links, joint_data, cm, output_file="robot.urdf", mesh_dir="", time_step=0):
    robot = ET.Element("robot", name="estimated_robot")
    pcds = []
    link_transforms = {}
    for i, link in enumerate(links):
        link_coords = [cm.coords[time_step][i] for i in link['cluster_idx']]
        link_pcds = [cm.clusters[time_step][str(i)] for i in link['cluster_idx']]
        link_matrices = [xyzquant2matrix_torch(coord).numpy() for coord in link_coords]
        avg_matrix = np.mean(link_matrices, axis=0)
        # mesh_filename = os.path.join(mesh_dir, f"{link['id']:04}.stl")
        # link_stl = o3d.io.read_triangle_mesh(mesh_filename)
        # link_stl.transform(avg_matrix)

        link_transforms[link['id']] = avg_matrix
        #pcds.append(link_stl)

    #o3d.visualization.draw_geometries(pcds)

    #print(link_transforms)
    link_pos_local = {}
    for joint in joint_data:
        child_frame = link_transforms[joint['child_link']]
        joint_frame = joint['global_pos']

        link_pos_local[joint['child_link']] = child_frame[:3, 3] - joint_frame[:3]
        print(f"Link {joint['child_link']} position in parent frame: {link_pos_local[joint['child_link']]}")

    # Create links
    color_map = plt.get_cmap("jet")
    colors = [color_map(i / len(links)) for i in range(len(links))]

    for link in links:
        link_name = f"link_{link['id']}"
        link_elem = ET.SubElement(robot, "link", name=link_name)
        
        # Get the transform for this link
        transform = link_transforms[link['id']]

        if link['parent_id'] is None:
            link_pos_local[link['id']] = transform[:3, 3]
        
        xyz = ' '.join(map(str, link_pos_local[link['id']])) # position
        rpy = ' '.join(map(str, np.zeros(3))) # orientation
        
        # Visual
        visual = ET.SubElement(link_elem, "visual")
        origin = ET.SubElement(visual, "origin", xyz=xyz, rpy=rpy)
        geometry = ET.SubElement(visual, "geometry")
        mesh_filename = os.path.join(mesh_dir, f"{link['id']:04}.stl")
        # mesh_filename = os.path.join(mesh_dir, f"{link['id']:04}.obj")
        ET.SubElement(geometry, "mesh", filename=mesh_filename, scale="1 1 1")

        # material
        material = ET.SubElement(visual, "material", name=f"material_{link['id']}")
        rgba = colors[link['id']][:3] + (1,) # add alpha channel
        # to string
        rgba = ' '.join(map(str, rgba))
        ET.SubElement(material, "color", rgba=rgba)
                
        # Collision (using the same mesh as visual)
        collision = ET.SubElement(link_elem, "collision")
        origin = ET.SubElement(collision, "origin", xyz=xyz, rpy=rpy)
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", filename=mesh_filename, scale="1 1 1")
        
        # Add inertial properties (example values, adjust as needed)
        inertial = ET.SubElement(link_elem, "inertial")
        inertial_origin = ET.SubElement(inertial, "origin", xyz=xyz, rpy=rpy)
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia", ixx="0.1", ixy="0.0", ixz="0.0", iyy="0.1", iyz="0.0", izz="0.1")
    
    # Create joints
    for joint in joint_data:
        joint_elem = ET.SubElement(robot, "joint", name=f"joint_{joint['child_link']}", type="revolute")
        
        parent_link = f"link_{joint['parent_link']}"
        ET.SubElement(joint_elem, "parent", link=parent_link)
        
        child_link = f"link_{joint['child_link']}"
        ET.SubElement(joint_elem, "child", link=child_link)
        
        # Calculate joint position and axis in parent frame
        parent_transform = link_transforms[joint['parent_link']]
        child_transform = link_transforms[joint['child_link']]
        
        # Joint position in parent frame
        global_pos = np.append(joint['global_pos'], 1)
        local_pos = np.linalg.inv(parent_transform) @ global_pos 
        origin_xyz = ' '.join(map(str, local_pos[:3] + link_pos_local[joint['parent_link']]))
        
        # Joint axis in parent frame
        global_axis = np.append(joint['global_axis'], 0)
        local_axis = np.linalg.inv(parent_transform[:3, :3]) @ global_axis[:3]
        local_axis = local_axis / np.linalg.norm(local_axis)
        
        # Calculate rotation from parent to child
        relative_rot = np.linalg.inv(parent_transform[:3, :3]) @ child_transform[:3, :3]
        relative_rot_euler = R.from_matrix(relative_rot).as_euler('xyz')
        origin_rpy = ' '.join(map(str, relative_rot_euler))
        
        ET.SubElement(joint_elem, "origin", xyz=origin_xyz, rpy=origin_rpy)
        ET.SubElement(joint_elem, "axis", xyz=' '.join(map(str, local_axis)))
        
        # Add some arbitrary limits
        limit = ET.SubElement(joint_elem, "limit", effort="100", velocity="100", lower="-3.14159", upper="3.14159")
    
    # Create the XML tree and save to file
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ", level=0)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"URDF file saved as {output_file}")

import pybullet as p
import pybullet_data
import math
import time

def visualize_urdf(urdf_path, ori = None):
    # Connect to the PyBullet physics server
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set up the simulation environment
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf", [0, 0, 0])
    
    # Load the URDF file
    startPos = [0, 0, 1.5]  # Lift the robot 1 unit above the ground
    if ori is None:
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    else:
        startOrientation = p.getQuaternionFromEuler(ori)
    robot = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True)
    
    # Get information about the loaded robot
    num_joints = p.getNumJoints(robot)
    print(f"Number of joints: {num_joints}")
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}")
    
    # Set up sliders for joint control
    sliders = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot, i)
        if joint_info[2] != p.JOINT_FIXED:
            slider = p.addUserDebugParameter(joint_info[1].decode('utf-8'), -math.pi, math.pi, 0)
            sliders.append((i, slider))
    
    # Main loop
    try:
        while True:
            # Update joint positions based on sliders
            for joint_index, slider in sliders:
                angle = p.readUserDebugParameter(slider)
                p.setJointMotorControl2(robot, joint_index, p.POSITION_CONTROL, targetPosition=angle)
            
            # Step the simulation
            p.stepSimulation()
            time.sleep(1./240.)  # Slow down the simulation
    
    except KeyboardInterrupt:
        print("Exiting...")
    
    # Disconnect from the physics server
    p.disconnect()
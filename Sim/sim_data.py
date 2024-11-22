import pybullet as p
import time
import pybullet_data
import math
import open3d as o3d
import numpy as np
import copy
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import json

class SimEnv:
    """
    Env class for the simulation environment
    1. Load the robot urdf
    2. Control the robot joints with position commands
    3. Simulate rgbd images from different camera poses
    """
    def __init__(
            self, 
            urdf_path: str, 
            base_position: list=[0, 0, 0], 
            base_orientation: list=[0, 0, 0], 
            gui: bool=False,
            dof: int=5,
            ground_flag: bool=False,
            radius: float=1.5,
            num_cameras: int=3,
            global_scale: float=1.0 # scale factor for urdf
        )->None:
        # Start the physics client
        self.gui = gui
        self.ground_flag = ground_flag
        self.dof = dof
        if gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self.ground_flag:
            self.floor_id = p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF(
            urdf_path, base_position, 
            p.getQuaternionFromEuler(base_orientation), 
            globalScaling = global_scale,
            # useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION) # add self collision flag
             useFixedBase=1)  # option in evalution, close self collision, allegro

        self.joint_params, self.joint_id_list = self._setup_joint_control()
        self.joint_list = list(self.joint_params.keys())
        self.dof_list = self.joint_list[:self.dof] 
        print('dof_list:', self.dof_list)
        self.joint_limits = np.array([self.joint_params[joint] for joint in self.dof_list]) # limits of the controled joints (in dof_list)
        # some joints are not used, give a fixed position, end effectors

        ## set the camera parameters
        self._setup_cameras(
            radius=radius, 
            num_cameras=num_cameras
        )

    def _setup_joint_control(self):
        # set a dictionary of joint names and their limits
        joint_params = {}
        joint_id_list = []
        num_joints = p.getNumJoints(self.robot_id)
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            # get joint type
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:
                joint_name = joint_info[1].decode('utf-8')
                joint_limit = [joint_info[8], joint_info[9]]
                joint_params[joint_name] = joint_limit

                joint_index = joint_info[0]
                joint_id_list.append(joint_index)
        return joint_params, joint_id_list
    

    def _setup_cameras(self, radius, num_cameras=20, cam_angle=20):
        """
        define camera positions with R, theta (aziuth), phi (elevation)
        cameras on a sphere with radius R, looking at the origin
        """
        cam_base = 0 # camera base height
        # sample theta by camera number
        if num_cameras < 20:
            cam_angles = np.array([cam_angle] * num_cameras)
            theta = np.linspace(0, 2 * np.pi, num_cameras, endpoint=False) # shape (num_cameras,)
            phi = np.pi * cam_angles / 180 #
            xs = radius * np.cos(theta) * np.cos(phi) # shape (num_cameras,)
            ys = radius * np.sin(theta) * np.cos(phi) # shape (num_cameras,)
            zs = radius * np.sin(phi) # shape (num_cameras,)

        else:
            # random sample theta, phi
            theta = np.random.rand(num_cameras) * 2 * np.pi
            phi = np.random.rand(num_cameras) * np.pi / 2
            xs = radius * np.cos(theta) * np.cos(phi)
            ys = radius * np.sin(theta) * np.cos(phi)
            zs = radius * np.sin(phi)

        self.cameras = [
            {'camera_pos': [x, y, z + cam_base],
             'target_pos': [0, 0,  cam_base], 
             'up_vector': [0, 0, 1], 
             'fov': 60, 'aspect': 1.0, 
             'near_val': 0.1, 
             'far_val': 4}

            for x, y, z in zip(xs, ys, zs)
        ]
    

    def reset(self):
        p.resetSimulation()
        p.disconnect(self.client_id)

    def step_simulation(self, timestep=1./240, n_step=600):
        # Simulation step progression, make sure the robot reaches the target position
        for _ in range(n_step):
            p.stepSimulation()
            if self.gui:
                time.sleep(timestep) # sleep when gui is on

    def set_joint_positions(self, commands, manual_positions=0):
        # set num_random joints to random positions, and set the rest to the given manual_positions

        # for j_id, joint_name in enumerate(self.joint_params.keys()):
        for j_id, joint_name in enumerate(self.joint_list):
            joint_limit = self.joint_params[joint_name] # [jointLowerLimit, jointUpperLimit]
            upper_limit = max(joint_limit[1], joint_limit[0])
            lower_limit = min(joint_limit[1], joint_limit[0])
            mid_point = (upper_limit + lower_limit) / 2
            # print(joint_name, joint_limit, upper_limit, lower_limit, mid_point)
            if joint_name in self.dof_list:
                # print(joint_name, commands[j_id])
                p.setJointMotorControl2(bodyIndex=self.robot_id, 
                                        jointIndex=self.joint_id_list[j_id], 
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=commands[j_id])
    
            else:
                fixed_position =  mid_point + manual_positions * (upper_limit - lower_limit) / 2 # set to the fixed position
                p.setJointMotorControl2(bodyIndex=self.robot_id, 
                                        jointIndex=self.joint_id_list[j_id], 
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=fixed_position)
                
        self.step_simulation()
        # get joint positions after step
        joint_positions = {}
        # for j_id, joint_name in enumerate(self.joint_params.keys()):
        for j_id, joint_name in enumerate(self.joint_list):
            joint_positions[joint_name] = p.getJointState(self.robot_id, self.joint_id_list[j_id])[0]  

        # print(joint_positions)    
        return joint_positions


    def render_camera(self, camera_index, width, height):
        # Render an image from a specific camera
        cam_info = self.cameras[camera_index]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_info['camera_pos'],
            cameraTargetPosition=cam_info['target_pos'],
            cameraUpVector=cam_info['up_vector']
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=cam_info['fov'],
            aspect=cam_info['aspect'],
            nearVal=cam_info['near_val'],
            farVal=cam_info['far_val']
        )
        
        width, height, rgb_pixels, depth_pixels, _ = p.getCameraImage(
            width, 
            height, 
            view_matrix, 
            proj_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        far = cam_info['far_val'] * 1000
        near = cam_info['near_val'] * 1000  ## attention!! open3d scale in mm, while pybullet scale in m
        depth_pixels = far * near / (far - (far - near) * depth_pixels)
        rgb_pixels = rgb_pixels[:, :, :3]
        # print(rgb_pixels.shape, depth_pixels.shape)
        rgb_pixels_contiguous = np.ascontiguousarray(rgb_pixels)
        rgb_image = o3d.geometry.Image(rgb_pixels_contiguous)
        depth_image = o3d.geometry.Image(depth_pixels)
        return view_matrix, proj_matrix, rgb_image, depth_image 
    

    def self_collision_check(self):
        # check self collision
        self_contact = p.getContactPoints(self.robot_id, self.robot_id)
        if self.ground_flag:
            floor_contact = p.getContactPoints(self.robot_id, self.floor_id)
        else:
            floor_contact = ()

        return self_contact, floor_contact

    def disable_collisions(self):
        link_name_to_index = {p.getJointInfo(self.robot_id, i)[12].decode('utf-8'): i for i in range(p.getNumJoints(self.robot_id))}
        
        for linkA_name, linkB_name in EXCLUDED_PARIS:
            linkA = link_name_to_index.get(linkA_name)
            linkB = link_name_to_index.get(linkB_name)
            
            if linkA is not None and linkB is not None:
                p.setCollisionFilterPair(self.robot_id, self.robot_id, linkA, linkB, enableCollision=0)
                # print(f"Collision disabled between {linkA_name} and {linkB_name}")


def transform_robot(robot_pcd, camera_pose_matrix):
    # transform robot pcd with the inverse camera pose matrix
    inverse_camera_matrix = np.linalg.inv(camera_pose_matrix)
    new_robot_pcd = copy.deepcopy(robot_pcd).transform(inverse_camera_matrix)
    return new_robot_pcd




def save_step_data(step_id: int, 
                   combined_pcds: o3d.geometry.PointCloud, 
                   joint_positions: dict, 
                   data_path: str,
                   dof_list: list
                   )->None:
    sub_path = data_path + f'{step_id:04}/'  # save to 4 decimal places
    os.makedirs(sub_path, exist_ok=True)
    o3d.io.write_point_cloud(sub_path + 'robot.ply', combined_pcds)
    with open(sub_path + 'joint_cfg.txt', 'w') as f:
        for joint_name, position in joint_positions.items():
            if joint_name in dof_list:
                f.write(f'{joint_name}:{position:,.6f}\n') # joint_name: str, position: float, save to 6 decimal places


def data_collection(env: SimEnv, 
                    data_path: str=None, 
                    width: int=800, 
                    height: int=800,
                    visualize: bool=False,
                    angle_list:np.array=None,
                    ground_flag: bool=False,
                    noise_flag: bool=False,
                    num_points: int=5000,
                    collision_flag: bool=False
                    )->None:
    """
    data collection loop
    1. get rgb and depth images from different camera poses
    2. convert to point clouds
    3. transform robot point clouds to world frames
    4. save the data
    """
    number_of_cameras = len(env.cameras) # get the number of cameras
    dof_list = env.dof_list
    collision = False
    noise = []

    if collision_flag:
        env.disable_collisions()

    pcds_record = []
        
    for jp_id in tqdm(range(len(angle_list))):

        joint_positions = env.set_joint_positions(commands=angle_list[jp_id]) # set joint positions
        self_c, floor_c = env.self_collision_check() # check self collision
        if len(self_c) + len(floor_c) > 0:
            print('collision detected', self_c, floor_c)
            collision = True
            break
            
        camera_poses = []
        pcds = []
        combined_pcds = o3d.geometry.PointCloud()

        for i in range(number_of_cameras):
            view_matrix, proj_matrix, rgb_image, depth_image = env.render_camera(i, width, height)
            view_matrix = np.reshape(view_matrix, (4, 4)).T
            camera_poses.append(view_matrix)

            # intrinsic matrix
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx= proj_matrix[0] * width / 2,  # Focal length x
                fy= proj_matrix[5] * height / 2,  # Focal length y

                cx= width / 2,  # Principal point x
                cy= height / 2)
            
            # generate RGBD image

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_trunc=4.0, convert_rgb_to_intensity=False)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    
            # flip, according to o3d documentation
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            if ground_flag:

                # Segment and Remove the plane
                _, inliers = pcd.segment_plane(distance_threshold=0.001,
                                                    ransac_n=6,
                                                    num_iterations=1000)
                ground = pcd.select_by_index(inliers)
                #o3d.visualization.draw_geometries([ground])
                robot_pcd = pcd.select_by_index(inliers, invert=True)
            else:
                robot_pcd = pcd

            pcds.append(robot_pcd)  # add robot pcds before transformation

        # add transformed robot pcds:
        for r_id, cam_p in enumerate(camera_poses):
            t_pcd = transform_robot(robot_pcd=pcds[r_id], camera_pose_matrix=cam_p)
            pcds.append(t_pcd)
            combined_pcds += t_pcd

        # whether to save the normalized joint positions
        save_position = joint_positions

        if noise_flag and jp_id != 0: # add noise to the data, not the first step

            # apply a position noise, to simulate the real world sacnning data
            pos_noise = np.random.normal(0, 0.01, size=3) # 3D noise
            noise.append(pos_noise)
            combined_pcds_save = copy.deepcopy(combined_pcds).translate(pos_noise) # add noise to the combined pcds

            # add noise to point clouds
            point_cloud_shape = np.array(combined_pcds_save.points).shape
            point_noise = np.random.normal(0, 0.0005, size=point_cloud_shape)
            combined_pcds_save.points = o3d.utility.Vector3dVector(np.array(combined_pcds_save.points) + point_noise)

            # downsample the point cloud
            combined_pcds_save = combined_pcds_save.farthest_point_down_sample(num_points)

        else:
            combined_pcds_save = combined_pcds.farthest_point_down_sample(num_points)

        # color to grey
        # combined_pcds_save.paint_uniform_color([0.3, 0.3, 0.3])


        if visualize:
            
            o3d.visualization.draw_geometries([combined_pcds, combined_pcds_save])

        if data_path != None:
            save_step_data(jp_id, combined_pcds_save, save_position, data_path, dof_list)

        pcds_record.append(combined_pcds_save)

    if noise_flag and data_path != None:
        np.savetxt(data_path + 'noise.txt', np.array(noise), fmt='%.6f')
    return collision, pcds_record


    

def angle_list(num_step, step_size, dof, joint_limits, scale, seed_i):
    """
    Generate a list of joint angles for data collection
    1. Randomly sample a set of target joint angles within the joint limits. (dof,)
    2. For each Joint, linearly interpolate the joint angles between start and target. (num_steps, dof)
    3. Update the start joint angles for the next epoch.

    Args:
    - num_step: int, number of steps
    - step_size: int, the max degree change for each step
    - dof: int, number of degrees of freedom
    - joint_limits: np.array(dof, 2), joint limits for each joint
    - scale: list, scale factor for joint limits

    Returns:
    - angle_list: np.array, (num_steps, dof)
    """
    start_rate = 0.5 # start from 30% of the limit
    low_step_limit = 0.2 # ensure target-start > 0.5 * abs_scaled_limit[j_id]

    np.random.seed(seed_i)
    joint_limits = joint_limits * 180 / np.pi # convert to degree
    scaled_limits = joint_limits * scale.reshape(-1, 1) # (dof, 2)
    abs_scaled_limit = np.abs(scaled_limits[:, 1] - scaled_limits[:, 0]) # (dof,)
    print('scaled_limits:', abs_scaled_limit)
    
    # start = np.zeros(dof) # start from 0
    # start from mid point
    # start = (scaled_limits[:, 0] + scaled_limits[:, 1]) / 2 # (dof,)
    
    start = scaled_limits[:, 0] + start_rate * (scaled_limits[:, 1] - scaled_limits[:, 0]) # start from 30% of the limit

    angle_list = []
    for j_id in range(dof):
        angle_list_i = []
        while len(angle_list_i) < num_step:
            # ensure target-start > 0.5 * abs_scaled_limit[j_id]
            while True:
                target = np.random.rand() * (scaled_limits[j_id][1] - scaled_limits[j_id][0]) + scaled_limits[j_id][0]
                if np.abs(target - start[j_id]) > low_step_limit * abs_scaled_limit[j_id]:
                # if np.abs(target - start[j_id]) > 3 * step_size:
                    break

            # add noise to the step size
            # step_size = step_size * (1 + np.random.rand() * 0.4 - 0.2) # if step_size=5, then 4-6
            step_size_rand = step_size * (1 + np.random.rand()) # if step_size=5, then 5-10
            
            num_steps = np.abs(target - start[j_id]) / step_size_rand
            num_steps = int(num_steps) + 1
            direction = 1 if target > start[j_id] else -1
            int_target = start[j_id] + direction * step_size_rand * num_steps

            angle_list_i += list(np.linspace(start[j_id], int_target, num_steps, endpoint=False)) # (num_steps,)
            start[j_id] = int_target
        angle_list.append(np.array(angle_list_i)[:num_step])

    angle_list = np.vstack(angle_list).T * np.pi / 180 # convert to radian

    return angle_list

def animate_raw_pcd(data_path, num_step):
    """
    Animate the raw PCD data from an angled view with fixed camera settings.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)

    # Read the first point cloud to initialize the visualizer
    pcd = o3d.io.read_point_cloud(data_path + f'{0:04}/robot.ply')
    vis.add_geometry(pcd)

    # Update the view control to fix the camera angle
    view_control = vis.get_view_control()
    view_control.set_front([0.5, -0.5, -1])  # Set to a diagonal view
    view_control.set_lookat([0, 0, 0])       # Look at the origin
    view_control.set_up([0, 0, 1])           # Z-axis is up
    view_control.set_zoom(0.8)               # Zoom to adjust the view

    # Loop through each point cloud and render it
    for i in range(num_step):
        pcd = o3d.io.read_point_cloud(data_path + f'{i:04}/robot.ply')

        vis.clear_geometries()  # Clear previous geometry
        vis.add_geometry(pcd)   # Add new point cloud

        vis.poll_events()
        vis.update_renderer()

        # Capture an image for each frame
        vis.capture_screen_image(data_path + f'{i:04}/robot.png')

    vis.destroy_window()

def collect():
    """
    loop through the angle list and collect data
    """
    seed = 0
    collision_free_seeds = []
    data_path_list = []

    while True:
        data_path = f"data/raw/{ROBOT}/{STEP_SIZE}_deg_{NUM_CAMERAS}_cams/V{seed:04}/"
        os.makedirs(data_path, exist_ok=True)

        robot_env = SimEnv(
            urdf_path = URDF_PATH, 
            base_orientation = ORIENTATION,
            gui = GUI,
            dof=DOF,
            ground_flag=GROUND,
            radius=CAM_DIST,
            num_cameras=NUM_CAMERAS)

        a_list = angle_list(
            num_step=NUM_STEP, 
            step_size=STEP_SIZE, 
            dof=DOF, 
            joint_limits=robot_env.joint_limits,
            scale=SCALE,
            seed_i=seed)
        
        print(a_list.shape)

        # plt.plot(a_list)
        # plt.show()

        # data collection loop
        collision, _ = data_collection(
            env = robot_env, 
            data_path = data_path, 
            width = WIDTH, 
            height = HEIGHT,
            visualize=VIS,
            angle_list=a_list,
            ground_flag=GROUND,
            noise_flag=NOISE,
            num_points=NUM_POINTS,
            collision_flag=COLLISION_EXCLUSION
        )

        robot_env.reset()

        if not collision:
            collision_free_seeds.append(int(seed))
            data_path_list.append(data_path)

        else:
            shutil.rmtree(data_path) # remove the data if collision detected

        if len(collision_free_seeds) >= EPOCHS:
            print('collision free seeds:', collision_free_seeds)
            # np.savetxt(f'data/raw/{ROBOT}/{DOF}dof_{WIDTH}pixel_seeds.txt', collision_free_seeds, fmt='%d')
            break

        seed += 1

    for data_path in data_path_list:
        animate_raw_pcd(data_path, NUM_STEP)

    p.disconnect()

if __name__ == "__main__":
    # Run at JointModel/, data saved at JointModel/data/, ignored by git
    # parameters

    # seed for numpy random
    np.random.seed(2024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='franka')
    parser.add_argument('--pix', type=int, default=800)
    parser.add_argument('--scale', type=float, default=0.9)
    parser.add_argument('--step_size', type=int, default=4)
    parser.add_argument('--num_step', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=5)  
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--ground', action='store_true')
    parser.add_argument('--no_noise', action='store_true')
    parser.add_argument('--num_points', type=int, default=5000)
    parser.add_argument('--num_cameras', type=int, default=20) # if num_cameras < 20, use fixed theta, phi, else random theta, phi
    args = parser.parse_args()
    # get json parameters by robot name
    with open('parameters.json') as f:
        params = json.load(f)
    robot_params = params[args.robot]

    """wx200"""
    WIDTH  = args.pix
    HEIGHT = args.pix
    ROBOT = args.robot # wx200, ur5, panda
    

    URDF_PATH = robot_params['gt']
    CAM_DIST = robot_params['cam_dist']
    ORIENTATION = robot_params['sim_ori']
    DOF = robot_params['dof']
    COLLISION_EXCLUSION = robot_params['collision_exclusion']
    EXCLUDED_PARIS = robot_params['excluded_pairs']
    # JOINTS = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate']  # 5 dof joints widowx200
    SCALE = np.array([args.scale] * DOF)
    GROUND = args.ground
    NOISE = not args.no_noise # add noise to the data
    NUM_POINTS = args.num_points
    NUM_CAMERAS = args.num_cameras

    NUM_STEP = args.num_step
    STEP_SIZE = args.step_size
    EPOCHS = args.epoch
    GUI = args.gui
    VIS = args.vis

    collect()


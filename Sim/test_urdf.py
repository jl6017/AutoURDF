import pybullet as p
import pybullet_data
import numpy as np
import math
import time
from PIL import Image
import argparse
import json
import os


def draw_coordinate_frame(position, orientation, axis_length=0.1):
    """
    Plots a coordinate frame in PyBullet at a given position and orientation.
    
    Parameters:
    - position: The position of the frame in the world (x, y, z).
    - orientation: The orientation of the frame as a quaternion (x, y, z, w).
    - axis_length: Length of the axis lines (default is 0.1).
    """
    # Convert the quaternion orientation to a rotation matrix
    rotation_matrix = p.getMatrixFromQuaternion(orientation)
    rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

    # Define the frame axes in the local frame (unit vectors for x, y, z)
    x_axis_local = np.array([1, 0, 0]) * axis_length
    y_axis_local = np.array([0, 1, 0]) * axis_length
    z_axis_local = np.array([0, 0, 1]) * axis_length

    # Transform the local axes to world coordinates
    x_axis_world = rotation_matrix @ x_axis_local
    y_axis_world = rotation_matrix @ y_axis_local
    z_axis_world = rotation_matrix @ z_axis_local

    # Define the end points of each axis in world coordinates
    x_end = position + x_axis_world
    y_end = position + y_axis_world
    z_end = position + z_axis_world

    # Plot the axes
    p.addUserDebugLine(position, x_end, [1, 0, 0], lineWidth=2)  # X-axis in red
    p.addUserDebugLine(position, y_end, [0, 1, 0], lineWidth=2)  # Y-axis in green
    p.addUserDebugLine(position, z_end, [0, 0, 1], lineWidth=2)  # Z-axis in blue



def visualize_urdf(urdf_path, ori = None):
    # Connect to the PyBullet physics server
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set up the simulation environment
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf", [0, 0, -1])
    view_params = p.getDebugVisualizerCamera()

    # Extract current yaw, pitch, and target position
    current_yaw = view_params[8]
    current_pitch = view_params[9]
    current_target = view_params[11]

    p.resetDebugVisualizerCamera(
        cameraDistance=CAM_DIST / 2,
        cameraYaw=current_yaw,
        cameraPitch=current_pitch,
        cameraTargetPosition=current_target
    )
        
    # Load the URDF file
    startPos = [0, 0, -0.2]  # Lift the robot 1 unit above the ground
    if ori is None:
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    else:
        startOrientation = p.getQuaternionFromEuler(ori)
    robot = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True, globalScaling=GOBAL_SCALE)
    
    # Get information about the loaded robot
    num_joints = p.getNumJoints(robot)
    print(f"Number of joints: {num_joints}")

    control = False
    if control:

        # record the link position and orientation in joint sequence
        link_pos_list = []
        link_ori_list = []
        for i in range(num_joints):
            # get link position and orientation
            link_state = p.getLinkState(robot, i)
            link_pos, link_ori = link_state[0], link_state[1]
            link_pos_list.append(link_pos)
            link_ori_list.append(link_ori) 
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}")

            # joint position and orientation in parent frame

            joint_uv, p_pos, p_ori, p_id = joint_info[-4:]

            print(f"Parent link: {p_id}, position: {p_pos}, orientation: {p_ori}") # in parent frame

            link_name = joint_info[12].decode('utf-8')
            print(f"Link name: {link_name}")

            if p_id == -1:
                link_pos, link_ori = p.getBasePositionAndOrientation(robot)
            else:
                link_pos, link_ori = link_pos_list[p_id], link_ori_list[p_id]

            print(f"Link position: {link_pos}, orientation: {link_ori}")

            # plot coordinate frame
            draw_coordinate_frame(link_pos, link_ori)

            rotation_matrix_l = np.array(p.getMatrixFromQuaternion(link_ori)).reshape(3, 3)
            rotation_matrix_j = np.array(p.getMatrixFromQuaternion(p_ori)).reshape(3, 3)
            rotation_matrix_world = rotation_matrix_l @ rotation_matrix_j.T
            joint_uv = np.array(joint_uv)
            joint_uv_world = rotation_matrix_world @ joint_uv
            p_pos = np.array(p_pos)
            p_pos_world = rotation_matrix_l @ p_pos

            # plot vector
            joint_pos = p_pos_world + np.array(link_pos)
            # link ori, joint ori, unit vector

            # draw_vector(joint_pos, joint_uv, [0, 0, 1])
            end = joint_pos + joint_uv_world * 0.5

            # Plot the vector

            p.addUserDebugLine(joint_pos, end, [0, 0, 1], lineWidth=5)  # X-axis in red


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

    else:
        # load camera parameters
        # Load view and projection matrices from .npy files
        # view_matrix = np.load(f"data/camera/{ROBOT}/view_matrix.npy")  # Replace with your .npy file path
        # projection_matrix = np.load(f"data/camera/{ROBOT}/proj_matrix.npy")  # Replace with your .npy file path

        # loop sin wave for all joints
        img_path = f'data/image/{ROBOT}/'
        os.makedirs(img_path, exist_ok=True)
        img_count = 0
        gif = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}")
            sin_step = 100
            sin_cmd = np.sin(np.linspace(0, 2*np.pi, sin_step)) * 1.57
            for j in range(sin_step):
                p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, targetPosition=sin_cmd[j])
                p.stepSimulation()
                time.sleep(1./24.)  # Slow down the simulation
                img_count += 1

                if img_count % 10 == 0:

                    # Save the GUI-rendered image
                    width, height, rgb_image, _, _ = p.getCameraImage(
                        width=view_params[0],   # Use the same width as the current GUI
                        height=view_params[1],  # Use the same height as the current GUI
                        renderer=p.ER_BULLET_HARDWARE_OPENGL
                    )

                    rgb_image = np.array(rgb_image, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
                    image = Image.fromarray(rgb_image)
                    gif.append(image)
                    image.save(f"{img_path}{img_count}.png")
        gif[0].save(f"{img_path}{ROBOT}.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)

                    
    # Disconnect from the physics server
    p.disconnect()


def gui_image():
    pass


if __name__ == "__main__":
    # urdf_path = "data/urdf/pxs_50_seg/4_deg_3_cams.urdf"
    # urdf_path = "data/urdf/bolt_30_seg/4_deg_20_cams.urdf"
    # urdf_path = "data/urdf/solo8_35_seg/4_deg_20_cams.urdf"
    # urdf_path = "data/urdf/pxs_45_seg/4_deg_20_cams.urdf"
    # urdf_path = "Robot/interbotix_xshexapod_descriptions/urdf/pxmark4s.urdf"
    # urdf_path = "data/urdf/solo8_35_seg/4_deg_20_cams.urdf"
    # urdf_path = "Robot/robot_properties_solo/resources/xacro/solo8.urdf"
    # urdf_path = "data/urdf/franka_20_seg/4_deg_20_cams.urdf"
    # urdf_path = "Robot/franka_panda/panda.urdf"
    
    # urdf_path = "Robot/bolt/bolt.urdf"

    # urdf_path = "data/urdf/op3_50_seg/4_deg_20_cams.urdf"
    # urdf_path = "Robot/ROBOTIS-OP3-Common-master/op3_description/op3_description/robotis_op3.urdf"
    # urdf_path = "data/urdf/allegro_30_seg/4_deg_20_cams.urdf"
    # urdf_path = "Robot/allegro_hand_description/allegro_hand_description_left_angle.urdf"

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='wx200_5')
    args = parser.parse_args()
    with open('parameters.json') as f:
        params = json.load(f)
    robot_params = params[args.robot]

    ROBOT = args.robot
    CAM_DIST = robot_params['cam_dist']
    NUM_SEG = robot_params['num_seg']


    urdf_path = f"data/urdf/{ROBOT}_{NUM_SEG}_seg/4_deg_20_cams.urdf"

    GOBAL_SCALE = 1 # for first allegro, op3
    visualize_urdf(urdf_path, ) # ori = [0, -0.785, 0.785]
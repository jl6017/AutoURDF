# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import time
from helper_functions import xyzquant2matrix_torch
from glob import glob
import pandas as pd
import seaborn as sns

def draw_silhouette_scores():
    # Dictionary to store all data
    all_data = []
    
    # Get list of robots
    robot_path = 'data/part/*/'
    robot_list = sorted(glob(robot_path))
    
    # Create two separate figures
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # List to store data for each group
    low_range_data = []
    high_range_data = []
    
    for robot in robot_list:
        robot_name = robot.split('/')[-2]
        print(f'Processing {robot_name}')
        
        # Get all version folders for this robot
        data_path = 'data/part/' + robot_name + '/*/'
        version_path = sorted(glob(data_path))[0]
        step_paths = sorted(glob(version_path + '*/'))
        
        scores_by_step = []
        robot_name_s = robot_name.split('_')[0]

        if robot_name.split('_')[1] == 'real':
            robot_name_s = robot_name.split('_')[0] + '_' + robot_name.split('_')[1]
        
        # Process each version
        for step_path in step_paths:
            print(f'Processing {step_path}')
            with open(step_path + 'score/silhouette_score.txt', 'r') as f:
                content = f.read()
                scores = eval(content.split('Score: ')[1].split('\n')[0])
                links_str = content.split('Links: ')[1].strip()
                n_links = [(int(x) - 1) for x in links_str.replace('[', '').replace(']', '').split()]
                scores_by_step.append(scores)
        
        scores_array = np.array(scores_by_step)
        mean_scores = np.mean(scores_array, axis=0)
        std_scores = np.std(scores_array, axis=0)
        
        # Store data in appropriate list based on robot name
        data_list = []
        for i, (mean, std, n_link) in enumerate(zip(mean_scores, std_scores, n_links)):
            data_list.append({
                'Robot': robot_name_s,
                'DoF': n_link,
                'Silhouette Score': mean,
                'Std': std
            })
            
        # Determine which group this robot belongs to
        if max(n_links) <= 8:  # Low range robots
            low_range_data.extend(data_list)
        else:  # High range robots
            high_range_data.extend(data_list)
    
    # Function to create plot for a group of robots
    def create_robot_plot(data, title, filename):
        plt.figure(figsize=(12, 6))
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Create the main plot
        sns.lineplot(data=df, 
                    x='DoF', 
                    y='Silhouette Score', 
                    hue='Robot',
                    marker='o')
        
        # Add error bands and peak points
        for robot_name in df['Robot'].unique():
            robot_data = df[df['Robot'] == robot_name]
            
            # Add error bands
            plt.fill_between(robot_data['DoF'],
                           robot_data['Silhouette Score'] - robot_data['Std'],
                           robot_data['Silhouette Score'] + robot_data['Std'],
                           alpha=0.2)
            
            # Find and mark the peak point
            peak_idx = robot_data['Silhouette Score'].idxmax()
            peak_x = robot_data.loc[peak_idx, 'DoF']
            peak_y = robot_data.loc[peak_idx, 'Silhouette Score']
            
            # Add star marker at peak
            plt.plot(peak_x, peak_y, '*', markersize=12, 
                    color=plt.gca().get_lines()[-1].get_color(),
                    label='_nolegend_')
        
        # Add a single "highest score" entry to legend
        plt.plot([], [], '*', markersize=10, color=plt.gca().get_lines()[-1].get_color(), label='Highest Score')
        
        # Customize the plot
        plt.title(title, pad=20)
        plt.xlabel('DoF')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        
        # Adjust legend
        plt.legend(title='Robot', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{filename}.svg', bbox_inches='tight')
        plt.close()
    
    # Create separate plots for each robot group
    create_robot_plot(low_range_data, 
                     'DoF Evaluation (Simple Robots)', 
                     'silhouette_scores_simple_robots')
    create_robot_plot(high_range_data, 
                     'DoF Evaluation (Complex Robots)', 
                     'silhouette_scores_complex_robots')
# def visualize_kinematic_tree_with_axes(cm, cluster_idx, kinematic_tree, joint_data, time_step=0):
#     # Set up color map
#     color_map = plt.get_cmap('jet')
#     colors = color_map(np.linspace(0, 1, len(cluster_idx)))[:,:3]

#     # reverse the color map
#     colors = colors[::-1]

#     color_seq = np.zeros(cm.num_coords)

#     for i, cluster in enumerate(cluster_idx):
#         color_seq[list(cluster)] = i

#     # Create point clouds for each cluster
#     pcds = []
#     cluster_pcds = cm.clusters[time_step]

#     matrices = [xyzquat_to_matrix_scipy(coord[:3], coord[3:]) for coord in cm.coords[time_step]]
#     for i in range(cm.num_coords):
#         cluster_np = cluster_pcds[str(i)]
#         cluster_np = cluster_np @ matrices[i][:3, :3].T + matrices[i][:3, 3]
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(cluster_np)
#         pcd.paint_uniform_color(colors[int(color_seq[i])])
#         pcds.append(pcd)

#     # Create cylinders for joint axes
#     vis = []
#     cy = True
#     if cy:
#         for parent, children in kinematic_tree.items():
#             parent_color = colors[parent]
#             for child in children:
#                 # Find the corresponding joint data
#                 joint = next(j for j in joint_data if j['clusters'] == (parent, child['child']) or j['clusters'] == (child['child'], parent))

#                 # get the global joint position and axis
#                 joint_ori= joint['global_axis']
#                 #print(joint_ori)
#                 joint_pos = joint['global_pos']

#                 # Create a cylinder
#                 cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=0.2)
#                 cylinder.compute_vertex_normals()
#                 cylinder.paint_uniform_color(parent_color)  # Same color as parent cluster link

#                 # Rotate the cylinder to align with the global axis
#                 cylinder_axis = np.array([0, 0, 1])
#                 rotation_axis = np.cross(cylinder_axis, joint_ori)
#                 rotation_angle = np.arccos(np.dot(cylinder_axis, joint_ori))
#                 R0 = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
#                 cylinder.rotate(R0, center=(0, 0, 0))

#                 # Position the cylinder
#                 cylinder.translate(joint_pos)

#                 vis.append(cylinder)

#     else:
#         for parent, children in kinematic_tree.items():
#             parent_color = colors[parent]
#             for child in children:
#                 # Find the corresponding joint data
#                 joint = next(j for j in joint_data if j['clusters'] == (parent, child['child']) or j['clusters'] == (child['child'], parent))

#                 # Get the global joint position and axis
#                 joint_axis = joint['global_axis']
#                 joint_pos = joint['global_pos']

#                 # Create a coordinate frame
#                 coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=joint_pos)

#                 # Rotate the coordinate frame to align Z-axis with the joint axis
#                 z_axis = np.array([0, 0, 1])
#                 rotation_axis = np.cross(z_axis, joint_axis)
#                 rotation_angle = np.arccos(np.dot(z_axis, joint_axis))
#                 R0 = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
#                 coord_frame.rotate(R0, center=joint_pos)

#                 # Color the coordinate frame (optional, as coordinate frames usually have standard colors)
#                 # If you want to keep a uniform color, you can uncomment the following line:
#                 # coord_frame.paint_uniform_color(parent_color)

#                 vis.append(coord_frame)

#     # Create coordinate frame
#     #coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

#     # Visualize
#     o3d.visualization.draw_geometries(pcds + vis)

def visualize_kinematic_tree(cm, links, joint_data, vis_time_step=0, scale=1, vis_params=None, robot=None):
    R_cylinder = 0.015 * scale
    H_cylinder = 0.2 * scale

    # Set up color map
    color_map = plt.get_cmap('jet')
    # colors = color_map(np.linspace(0, 1, len(links)))[:,:3]
    colors = [color_map(i / len(links)) for i in range(len(links))]
    colors = [c[:3] for c in colors]
    color_seq = np.zeros(cm.num_coords)

    for link in links:
        for coord in link['cluster_idx']:
            color_seq[coord] = link['id']

    # Create point clouds for each cluster
    time_step = 0
    vis_pcds = None
    while time_step <= 9:

        pcds = o3d.geometry.PointCloud()
        cluster_pcds = cm.clusters[time_step]

        matrices = [xyzquant2matrix_torch(coord).numpy() for coord in cm.coords[time_step]]
        matrices = np.array(matrices)
        for i in range(cm.num_coords):
            cluster_np = cluster_pcds[str(i)]
            cluster_np = cluster_np @ matrices[i][:3, :3].T + matrices[i][:3, 3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster_np)
            pcd.paint_uniform_color(colors[int(color_seq[i])])
            pcds += pcd

        #save to ply
        # eval_path = 'data/evaluation/eva_data/ours/'
        # os.makedirs(eval_path, exist_ok=True)
        # o3d.io.write_point_cloud(f'data/evaluation/eva_data/ours/{time_step}.ply', pcds)

        if time_step == vis_time_step:
            vis_pcds = pcds
        time_step += 1
    # Create cylinders for joint axes
    cylinder_joint = []
    for joint in joint_data:
        parent_link = next(link for link in links if link['id'] == joint['parent_link'])
        child_link = next(link for link in links if link['id'] == joint['child_link'])
        parent_color = colors[parent_link['id']]
        # Get the global joint position and axis
        joint_axis = joint['global_axis']
        joint_pos = joint['global_pos']

        # Create a cylinder
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=R_cylinder, height=H_cylinder)
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(parent_color)

        # Rotate the cylinder to align with the global axis
        cylinder_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(cylinder_axis, joint_axis)
        rotation_angle = np.arccos(np.dot(cylinder_axis, joint_axis))
        R0 = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        cylinder.rotate(R0, center=(0, 0, 0))

        # Position the cylinder
        cylinder.translate(joint_pos)

        cylinder_joint.append(cylinder)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=844, height=773)
        
    # Add geometries
    for geom in cylinder_joint:
        vis.add_geometry(geom)

    # add point cloud
    vis.add_geometry(vis_pcds)
        
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
    vis.capture_screen_image(f"data/image/{robot}_{int(time.time())}.png", do_render=True)
    # Close the window
    vis.destroy_window()

def plot_coordinate_frame(ax, origin, R, length=0.05, o_color=np.array([1, 0, 0])):
    """
    Plots a coordinate frame on the given axes.
    
    Parameters:
    - ax: The matplotlib Axes3D object to plot on.
    - origin: The origin of the coordinate frame as a 3D vector [x, y, z].
    - R: The rotation matrix representing the orientation of the frame.
    - length: The length of the frame axes.
    """
    # Define the unit vectors for the coordinate frame
    x_axis = R[:, 0] * length  # x-axis direction
    y_axis = R[:, 1] * length  # y-axis direction
    z_axis = R[:, 2] * length  # z-axis direction
    
    # Plot the origin
    ax.scatter(*origin, color=o_color, s=50)
    
    # Plot the axes using quiver
    ax.quiver(*origin, *x_axis, color='r', arrow_length_ratio=0.5, label='x-axis')
    ax.quiver(*origin, *y_axis, color='g', arrow_length_ratio=0.5, label='y-axis')
    ax.quiver(*origin, *z_axis, color='b', arrow_length_ratio=0.5, label='z-axis')


def rpy_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw to a rotation matrix.
    """
    # Compute individual rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def coordinate_animation(coords):
    """
    Animate the coordinate map
    Args:
    - coords: cm.coords: (time-step, num_points, 6), np array
    """
    data_shape = coords.shape
    colors = np.random.rand(data_shape[1], 3)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 5])
    # ax.set_zlim([0, 5])
    for _ in range(10):
        for i in range(data_shape[0]):
            ax.clear()
            ax.grid(False)
            ax.set_xlim([-0.4, 0.4])
            ax.set_ylim([-0.4, 0.4])
            ax.set_zlim([-0., 0.5])
            for j in range(data_shape[1]):
                origin = coords[i, j, :3]
                R = rpy_to_rotation_matrix(*coords[i, j, 3:])
                plot_coordinate_frame(ax, origin, R, o_color=colors[j])
            plt.pause(0.05)
    plt.show()

def matrix_animation(matrices):
    """
    Animate the coordinate map
    Args:
    - matrices: (time-step, num_points, 4, 4), np array
    """
    data_shape = matrices.shape
    colors = np.random.rand(data_shape[1], 3)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 5])
    # ax.set_zlim([0, 5])
    for _ in range(10):
        for i in range(data_shape[0]):
            ax.clear()
            ax.grid(False)
            ax.set_xlim([-0.4, 0.4])
            ax.set_ylim([-0.4, 0.4])
            ax.set_zlim([-0., 0.5])
            for j in range(data_shape[1]):
                origin = matrices[i, j, :3, 3]
                R = matrices[i, j, :3, :3]
                plot_coordinate_frame(ax, origin, R, o_color=colors[j])
            plt.pause(0.5)
    plt.show()


if __name__ == "__main__":
    # data_path = 'PointCloud/temp/data_20seg_400p_2023s/'
    # cm = CoordMap(data_path) # cm.coords: (time-step, num_points, 6), np array
    # coordinate_animation(cm.coords)

    # data_path = 'PointCloud/temp(0)_allergo/data_30seg_800p_7s/'
    # import glob

    # files = sorted(glob.glob(data_path + 'matrix/*.npy'))
    # matrices = []
    # for file in files:
    #     matrices.append(np.load(file))
    # matrices = np.array(matrices)
    # print(matrices.shape)

    # matrix_animation(matrices)
    draw_silhouette_scores()
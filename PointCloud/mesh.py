import pyvista as pv
import numpy as np
import glob
import matplotlib.pyplot as plt
import open3d as o3d
import argparse
import json

def plot_mesh(link_path, step, save_pc=False):
    """
    Plot the mesh
    Args:
    - link_path: str, path to the link
    - step: int, step index
    """
    matrix = np.load(link_path + f'matrix/{step:04}.npy')
    mesh_path = sorted(glob.glob(link_path + '*.stl'))

    # pyvista plot
    # mesh = [pv.read(path) for path in mesh_path]
    # mesh = [m.transform(matrix[i]) for i, m in enumerate(mesh)]
    # p = pv.Plotter()
    # colormap = plt.get_cmap("jet")
    # colors = [colormap(i / DOF+1) for i in range(DOF+1)]
    # for i, m in enumerate(mesh):
    #     p.add_mesh(m, color=colors[i], show_edges=False)
    # p.show()

    # open3d plot
    mesh = [o3d.io.read_triangle_mesh(path) for path in mesh_path]
    mesh = [m.transform(matrix[i]) for i, m in enumerate(mesh)]
    # compute normals
    for m in mesh:
        m.compute_vertex_normals()
    colormap = plt.get_cmap("jet")
    colors = [colormap(i / (DOF+1)) for i in range(DOF+1)]
    if save_pc:
        pc_path = sorted(glob.glob(link_path + '*.ply'))
        # exclue file with '_og' in the name
        pc_path = [path for path in pc_path if '_og' not in path]
        pc = [o3d.io.read_point_cloud(path) for path in pc_path]
        pc = [p.transform(matrix[i]) for i, p in enumerate(pc)]
        merged = o3d.geometry.PointCloud()
        for i, p in enumerate(pc):
            p.paint_uniform_color(colors[i][:3])
            merged += p
        # save merged point cloud
        # create folder if not exist
        import os
        if not os.path.exists(link_path +'merged/'):
            os.makedirs(link_path +'merged/')
        o3d.io.write_point_cloud(link_path +'merged/'+ f'{step}.ply', merged)
    else:
        for i, m in enumerate(mesh):
            m.paint_uniform_color(colors[i][:3])
        o3d.visualization.draw_geometries(mesh)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='wx200_4')
    parser.add_argument('--num_cameras', type=int, default=3) # number of cameras
    parser.add_argument('--step_size', type=int, default=5) # motor step size
    args = parser.parse_args()

    # get json parameters by robot name
    with open('parameters.json') as f:
        params = json.load(f)
    robot_params = params[args.robot]

    ROBOT = args.robot
    NUM_SEG = robot_params['num_seg']
    DOF = robot_params['dof']
    STEP_SZIE = args.step_size
    NUM_CAMERAS = args.num_cameras

    LINK_PATHS = f'data/mesh/{ROBOT}_{NUM_SEG}_seg/{STEP_SZIE}_deg_{NUM_CAMERAS}_cams/*/'

    for LINK_PATH in sorted(glob.glob(LINK_PATHS)):
        print(LINK_PATH)
        for step in range(0, 20):
            plot_mesh(LINK_PATH, step=step, save_pc=True)

        plot_mesh(LINK_PATH, step=19, save_pc=False)  # TODO: step 9, incorrect coordinate


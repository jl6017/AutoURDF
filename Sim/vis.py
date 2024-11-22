import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.animation import FuncAnimation, PillowWriter
import glob

def animate_point_cloud_sequence(ply_files, save_as=None):
    """
    Animate a sequence of point clouds using Open3D for loading and Matplotlib for visualization.

    Parameters:
        ply_files (list of str): List of paths to the .ply files.
        save_as (str): Path to save the animation as a video file (optional, e.g., 'animation.mp4').
    """
    # Load the first point cloud to initialize the plot
    pcd = o3d.io.read_point_cloud(ply_files[0])
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # Set up the Matplotlib figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    plt.tight_layout()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first frame
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=colors if colors is not None else 'gray', s=5)

    # Set axis labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point Cloud Animation")

    # Determine the axis limits from all point clouds 
    all_points = np.vstack([np.asarray(o3d.io.read_point_cloud(f).points) for f in ply_files])
    # max abs
    x_lim = (-np.max(np.abs(all_points[:, 0])), np.max(np.abs(all_points[:, 0])))
    y_lim = (-np.max(np.abs(all_points[:, 1])), np.max(np.abs(all_points[:, 1])))
    z_lim = (0, np.max(np.abs(all_points[:, 2])))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    # no grid
    # ax.grid(False)
    # no axis
    # ax.axis("off")  # Turn off axes (ticks, labels, and grid)

    def update(frame):
        """Update function for the animation."""
        # Load the current point cloud
        pcd = o3d.io.read_point_cloud(ply_files[frame])
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # Update the scatter plot data
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        if colors is not None:
            scatter.set_color(colors)

        ax.set_title(f"Point Cloud Animation - Frame {frame + 1}/{len(ply_files)}")
        return scatter,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(ply_files), interval=100, blit=False)

    # Save the animation if requested
    if save_as:
        # ani.save(save_as, fps=1, extra_args=['-vcodec', 'libx264'])
        # print(f"Animation saved as {save_as}")
        gif_writer = PillowWriter(fps=2)
        ani.save(save_as, writer=gif_writer)
        print(f"Animation saved as {save_as}")

    # Show the animation
    plt.show()

if __name__ == "__main__":
    # Example usage
    # List of paths to your .ply files
    path_list = sorted(glob.glob("data/raw/franka/4_deg_20_cams/V0000/*/robot.ply"))

    animate_point_cloud_sequence(path_list, save_as="point_cloud_animation.gif")

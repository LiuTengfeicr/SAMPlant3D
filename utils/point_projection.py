import os

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def projection_xy(pcd, sampic, filename):
    point = np.asarray(pcd.points)
    # points_2d = point[:, :2]
    points_2d = point[:, [0, 2]]

    plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.gca()
    ax.set_facecolor((0, 0, 0, 0))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, c='blue')

    plt.axis('off')

    file_path = os.path.join(sampic, f'{filename}.png')
    plt.savefig(file_path, pad_inches=0, transparent=True, dpi=100)
    plt.close()

    return file_path

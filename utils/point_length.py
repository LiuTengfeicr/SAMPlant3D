import numpy as np
import open3d as o3d
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt


def plate_scale(pcd):
    point = np.asarray(pcd.points)
    points_2d = point[:, [0, 2]]

    points_shapely = MultiPoint(points_2d)

    min_rect = points_shapely.minimum_rotated_rectangle

    rect_x, rect_y = min_rect.exterior.xy

    plt.figure(figsize=(10, 10))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, c='blue', label='Point Cloud')
    plt.plot(rect_x, rect_y, 'r-', label='Minimum Rotated Rectangle')

    x, y = min_rect.exterior.xy
    distances = [np.linalg.norm(np.array([x[i], y[i]]) - np.array([x[i + 1], y[i + 1]])) for i in range(len(x) - 1)]

    length = distances[0]
    width = distances[1]

    a = [length, width]

    l = np.min(a)
    scale = 25 / l

    return l, scale

import numpy as np
import open3d as o3d
import math


def rotate_point(pcd):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    [a, b, c, d] = plane_model
    normal_vector = np.array([a, b, c])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    z_axis = np.array([0, 0, 1])

    rotation_axis = np.cross(normal_vector, z_axis)
    rotation_angle = np.arccos(np.clip(np.dot(normal_vector, z_axis), -1.0, 1.0))

    if np.linalg.norm(rotation_axis) < 1e-6:
        rotation_matrix = np.identity(3)
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        axis_angle = rotation_axis * rotation_angle
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
        pcd.rotate(rotation_matrix)

    x_axis_rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    pcd.rotate(x_axis_rotation_matrix)

    final_rotation_matrix = x_axis_rotation_matrix @ rotation_matrix

    return pcd, final_rotation_matrix

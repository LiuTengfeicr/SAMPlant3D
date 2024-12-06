import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA


def compute_alpha_shape_volume(pcd, alpha, step=0.1, max_alpha=5.0):
    while alpha <= max_alpha:
        try:

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            mesh.compute_vertex_normals()

            # 尝试计算体积
            num = mesh.get_volume()
            return num
        except RuntimeError:
            print(f"Alpha value {alpha} too low, increasing alpha...")
            alpha += step

    print("Failed to compute volume even with maximum alpha value.")
    return 0


def calculate_normalized_volume(points, normalization_factor=1.0):
    points = np.asarray(points.points)
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    normalized_volume = volume / normalization_factor
    return normalized_volume, x_max - x_min, y_max - y_min, z_max - z_min


def hight(pcd, scale):
    points = np.asarray(pcd.points)
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)

    lengths = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    pcd_max = scale * np.max(lengths)
    pcd_min = scale * np.min(lengths)

    sorted_lengths = np.sort(lengths)
    pcd_mid = scale * sorted_lengths[1]

    return pcd_max, pcd_min, pcd_mid


def compute_surface_area(pcd, alpha, step=0.1, max_alpha=5.0):
    while alpha <= max_alpha:
        try:

            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
            alpha_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
            alpha_shape.compute_vertex_normals()

            surface_area = alpha_shape.get_surface_area()
            return surface_area
        except RuntimeError:
            print(f"Alpha value {alpha} too low, increasing alpha...")
            alpha += step

    print("Failed to compute surface area even with maximum alpha value.")
    return 0


def calculate_vectors_angle(pcd):
    points = np.asarray(pcd.points)

    y_median = np.median(points[:, 1])
    lower_half = points[points[:, 1] <= y_median]
    upper_half = points[points[:, 1] > y_median]

    pca = PCA(n_components=3)
    pca.fit(lower_half)
    trend_vector = pca.components_[0]

    y_upper_median = np.median(upper_half[:, 1])
    upper_upper_half = upper_half[upper_half[:, 1] > y_upper_median]
    lower_upper_half = upper_half[upper_half[:, 1] <= y_upper_median]

    upper_upper_half_pcd = o3d.geometry.PointCloud()
    upper_upper_half_pcd.points = o3d.utility.Vector3dVector(upper_upper_half)

    plane_model, inliers = upper_upper_half_pcd.segment_plane(
        distance_threshold=0.1,
        ransac_n=3,
        num_iterations=1000
    )

    [a, b, c, d] = plane_model
    normal_vector = np.array([a, b, c])
    normal_vector /= np.linalg.norm(normal_vector)

    y_axis = np.array([0, 1, 0])
    cos_theta_normal = np.dot(normal_vector, y_axis) / (np.linalg.norm(normal_vector) * np.linalg.norm(y_axis))
    angle_with_y_axis = np.arccos(cos_theta_normal)

    if angle_with_y_axis > np.pi / 2:
        angle_with_y_axis = np.pi - angle_with_y_axis
    if angle_with_y_axis == np.pi / 2:
        angle_with_y_axis = 0

    cos_theta_trend_normal = np.dot(trend_vector, normal_vector) / (
            np.linalg.norm(trend_vector) * np.linalg.norm(normal_vector))
    angle_trend_normal = np.arccos(cos_theta_trend_normal)

    if angle_trend_normal > np.pi / 2:
        angle_trend_normal = np.pi - angle_trend_normal
    angle_trend_normal = np.pi / 2 - angle_trend_normal

    return trend_vector, normal_vector, np.degrees(angle_with_y_axis), np.degrees(angle_trend_normal), points


def calculate_leaf_angle(leaf, stem, min_points_threshold=3):
    """
     Computes the normal vector of the leaf and the trend vector of the stem.
    If the number of points in the upper half of the leaf is less than min_points_threshold, the normal vector is calculated using the whole leaf.

    Parameters:
    - leaf: Open3D point cloud object, represents the leaf part.
    - stem: Open3D point object, represents the stem part.
    - min_points_threshold: the threshold of the minimum number of points to be used for determining whether to use the upper half of the point cloud.

    return:
    - trend_vector: trend vector of stem
    - normal_vector: normal vector of leaf
    - angle_with_y_axis: angle of normal vector with y-axis (in degrees)
    - angle_trend_normal: angle between trend vector and normal vector (in degrees)
    """

    stem_points = np.asarray(stem.points)
    stem_center = np.mean(stem_points, axis=0)
    stem_trend_vector = stem_center / np.linalg.norm(stem_center)

    leaf_points = np.asarray(leaf.points)

    y_median = np.median(leaf_points[:, 1])

    upper_half_indices = np.where(leaf_points[:, 1] > y_median)[0]
    upper_half_points = leaf_points[upper_half_indices]

    if len(upper_half_points) >= min_points_threshold:

        upper_half_leaf = o3d.geometry.PointCloud()
        upper_half_leaf.points = o3d.utility.Vector3dVector(upper_half_points)

        plane_model, inliers = upper_half_leaf.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=1000
        )
    else:

        print("Insufficient number of point clouds in the upper half, use the whole leaf for calculations.")
        plane_model, inliers = leaf.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=1000
        )

    [a, b, c, d] = plane_model
    leaf_normal_vector = np.array([a, b, c])
    leaf_normal_vector /= np.linalg.norm(leaf_normal_vector)

    y_axis = np.array([0, 1, 0])
    cos_theta_normal = np.dot(leaf_normal_vector, y_axis)
    angle_with_y_axis = np.arccos(cos_theta_normal / (np.linalg.norm(leaf_normal_vector) * np.linalg.norm(y_axis)))

    cos_theta_trend_normal = np.dot(stem_trend_vector, leaf_normal_vector)
    angle_trend_normal = np.arccos(
        cos_theta_trend_normal / (np.linalg.norm(stem_trend_vector) * np.linalg.norm(leaf_normal_vector))
    )

    if angle_with_y_axis > np.pi / 2:
        angle_with_y_axis = np.pi - angle_with_y_axis
    if angle_trend_normal > np.pi / 2:
        angle_trend_normal = np.pi - angle_trend_normal

    return stem_trend_vector, leaf_normal_vector, np.degrees(angle_with_y_axis), np.degrees(angle_trend_normal)

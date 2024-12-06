import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RANSACRegressor


def read_point_cloud(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()[:3]
            try:
                points.append([float(v) for v in values])
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
                continue
    points_array = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    return pcd


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

        print("The upper half of the point cloud is insufficient in number and is computed using the entire leaf.")
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


def visualize_pcd_with_vectors(pcd):
    trend_vector, normal_vector, angle_with_y_axis, angle_trend_normal, points = calculate_vectors_angle(pcd)

    scale_factor = 1
    scaled_points = points.copy()
    scaled_points[:, 1] *= scale_factor

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(scaled_points[:, 0], scaled_points[:, 2], scaled_points[:, 1], s=1, color='b', label='Point Cloud')

    intersection_point = np.mean(scaled_points, axis=0)

    ax.quiver(intersection_point[0], intersection_point[2], intersection_point[1],
              trend_vector[0], trend_vector[2], trend_vector[1] * scale_factor,
              color='g', label='Stem Vector', length=0.1, normalize=True)

    ax.quiver(intersection_point[0], intersection_point[2], intersection_point[1],
              normal_vector[0], normal_vector[2], normal_vector[1] * scale_factor,
              color='r', label='Leaf Normal Vector', length=0.1, normalize=True)

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_zlim(-1.8, -1.2)

    ax.set_title(f'Leaf Angle: {angle_trend_normal:.3f}')

    plt.show()


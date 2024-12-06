import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm


def extract_xyz_between_one_fourth_and_one_third_y(points):
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    y_one_third = y_min + (y_max - y_min) / 3
    y_one_fourth = y_min + (y_max - y_min) / 4
    between_one_fourth_and_one_third_y_points = points[(points[:, 1] > y_one_fourth) & (points[:, 1] <= y_one_third)]
    xz_points = between_one_fourth_and_one_third_y_points[:, [0, 2]]
    return between_one_fourth_and_one_third_y_points, xz_points, y_one_fourth, y_one_third


def detect_clusters(xz_points, eps=0.05, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xz_points)
    labels = clustering.labels_
    return labels


def find_all_clusters(xz_points, labels, unique_labels):
    cluster_centers = []
    cluster_radii = []

    for cluster in unique_labels:
        if cluster == -1:  # 跳过噪声点
            continue
        cluster_points = xz_points[labels == cluster]
        center = cluster_points.mean(axis=0)
        radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
        cluster_centers.append(center)
        cluster_radii.append(radius)

    return unique_labels[unique_labels >= 0], cluster_centers, cluster_radii


def assign_points_to_clusters(points, clusters, cluster_centers, cluster_radii):
    labels = np.full(len(points), -1)

    for i, point in enumerate(points):
        distances = [np.linalg.norm(point[[0, 2]] - center) for center in cluster_centers]
        labels[i] = clusters[np.argmin(distances)]

    return labels


def point_sm(pcd):
    points = pcd
    unique_labels = np.unique(points[:, -1])
    new_labels = points[:, -1].copy()

    max_existing_label = int(np.max(unique_labels))
    next_label = max_existing_label + 1

    for label in tqdm(unique_labels):
        label_points = points[points[:, -1] == label]

        between_one_fourth_and_one_third_y_points, xz_points, y_one_fourth, y_one_third = extract_xyz_between_one_fourth_and_one_third_y(
            label_points)
        if xz_points.shape[0] == 0:
            continue
        labels = detect_clusters(xz_points)

        unique_labels1, counts = np.unique(labels[labels >= 0], return_counts=True)

        if len(unique_labels1) >= 2:
            clusters, cluster_centers, cluster_radii = find_all_clusters(xz_points, labels, unique_labels1)

            final_labels = assign_points_to_clusters(label_points, clusters, cluster_centers, cluster_radii)

            for j, point in enumerate(label_points):
                idx = np.where((points == point).all(axis=1))[0][0]
                if final_labels[j] != -1:
                    new_labels[idx] = next_label + final_labels[j]

            next_label += len(np.unique(final_labels[final_labels != -1]))

    updated_pcd = np.hstack((points[:, :3], new_labels.reshape(-1, 1)))
    return updated_pcd

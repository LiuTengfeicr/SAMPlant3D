import numpy as np
import open3d as o3d


def farthest_point_sampling(pcd, num_clusters):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    centroids = [np.random.randint(len(pcd.points))]
    points = np.asarray(pcd.points)
    distances = np.full(len(pcd.points), np.inf)

    for _ in range(num_clusters - 1):
        farthest_dist = np.argmax(distances)
        centroids.append(farthest_dist)
        new_distances = np.linalg.norm(points - points[farthest_dist], axis=1)
        distances = np.minimum(distances, new_distances)

    clusters = {i: [] for i in centroids}
    for i, point in enumerate(points):
        idx = min(clusters, key=lambda x: np.linalg.norm(point - points[x]))
        clusters[idx].append(i)

    return clusters


def remove_noise_with_statistical_outlier(pcd, nb_neighbors=20, std_ratio=2.0):
    clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    return clean_pcd


def remove_noise(tray, plant_point, plant):
    if tray:
        tray = remove_noise_with_statistical_outlier(tray, nb_neighbors=100, std_ratio=2.0)
    elif plant_point:
        # plant_point = farthest_point_sampling(plant_point, num_clusters=10)
        plant_point = remove_noise_with_statistical_outlier(plant_point, nb_neighbors=50, std_ratio=2.0)
    elif plant:
        plant = remove_noise_with_statistical_outlier(plant, nb_neighbors=100, std_ratio=1.0)
    return tray, plant_point, plant

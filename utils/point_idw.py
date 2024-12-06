import numpy as np
import open3d as o3d
from scipy.spatial import distance_matrix, KDTree
from point_noise import remove_noise_with_statistical_outlier


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


def idw_interpolation(points, num_points):
    new_positions = np.random.rand(num_points, 3) * (points.max(axis=0) - points.min(axis=0)) + points.min(axis=0)
    interpolated_points = []
    for new_pos in new_positions:
        distances = np.linalg.norm(points - new_pos, axis=1)
        weights = 1 / (distances ** 2 + 1e-8)
        weighted_sum = np.dot(weights, points)
        interpolated_points.append(weighted_sum / weights.sum())
    return np.array(interpolated_points)


def idw_interpolation1(points, new_positions, power=2):
    num_points = new_positions.shape[0]
    interpolated_points = np.zeros((num_points, 3))
    for i in range(num_points):
        distances = np.linalg.norm(points - new_positions[i], axis=1)
        weights = 1 / (distances ** power + 1e-8)
        weighted_sum = np.dot(weights, points) / np.sum(weights)
        interpolated_points[i] = weighted_sum

    return interpolated_points


def apply_gaussian_noise(points, std_dev=0.01):
    noise = np.random.normal(0, std_dev, points.shape)
    noisy_points = points + noise
    return noisy_points


def process_clusters(pcd):
    num_clusters = 10
    clusters = farthest_point_sampling(pcd, num_clusters)
    enhanced_points = []

    for centroid, indices in clusters.items():
        cluster_points = np.asarray(pcd.points)[indices]
        if len(indices) < 30:
            num_interpolations = 1000
        elif 30 <= len(indices) < 100:
            num_interpolations = 200
        else:
            num_interpolations = 50

        interpolated_points = idw_interpolation(cluster_points, num_interpolations)

        noisy_original_points = apply_gaussian_noise(cluster_points)
        noisy_interpolated_points = apply_gaussian_noise(interpolated_points)

        enhanced_points.extend(noisy_original_points)
        enhanced_points.extend(noisy_interpolated_points)

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(np.array(enhanced_points))
    return new_pcd


def linear_interpolate_points(p1, p2, num_points=10):
    return np.array([p1 + i / num_points * (p2 - p1) for i in range(1, num_points)])


def connect_breaks_in_point_cloud(pcd, p1, p2):
    # 生成新点
    new_points = linear_interpolate_points(p1, p2, num_points=40)

    # 对新点加入高斯扰动
    noisy_new_points = apply_gaussian_noise(new_points, std_dev=0.02)

    # 添加到现有点云中
    extended_points = np.vstack((np.asarray(pcd.points), noisy_new_points))
    pcd.points = o3d.utility.Vector3dVector(extended_points)
    return pcd


def linear_interpolation_half_segment(start, end):
    num_points = 50
    points = np.linspace(start, end, num_points + 1)[(num_points // 2 + 1):]
    return points


def cluster_DBSCAN(pcd, points, eps1, eps2):
    labels_small = np.array(pcd.cluster_dbscan(eps1, min_points=10, print_progress=False))
    labels_large = np.array(pcd.cluster_dbscan(eps2, min_points=10, print_progress=False))

    clusters_small = {i: points[labels_small == i] for i in range(labels_small.max() + 1)}
    clusters_large = {i: points[labels_large == i] for i in range(labels_large.max() + 1)}

    centroids_small = {i: np.mean(clusters_small[i], axis=0) for i in clusters_small}
    centroids_large = {i: np.mean(clusters_large[i], axis=0) for i in clusters_large}

    cluster_mapping = {}
    for i in centroids_small:
        point = centroids_small[i]
        for j in centroids_large:
            if np.array_equal(point, centroids_large[j]) or np.linalg.norm(point - centroids_large[j]) < eps2:
                if j not in cluster_mapping:
                    cluster_mapping[j] = []
                cluster_mapping[j].append(i)
                break

    for large_cluster, small_clusters in cluster_mapping.items():
        for i in range(len(small_clusters) - 1):
            p1 = centroids_small[small_clusters[i]]
            p2 = centroids_small[small_clusters[i + 1]]
            pcd = connect_breaks_in_point_cloud(pcd, p1, p2)

    kdtrees = {i: KDTree(clusters_large[i]) for i in clusters_large}
    enhanced_points = np.asarray(pcd.points)

    for i in centroids_large:
        for j in centroids_large:
            if i != j:
                centroid = centroids_large[i].reshape(1, -1)
                dist, idx = kdtrees[j].query(centroid, k=1)
                nearest_point = clusters_large[j][idx[0]]

                interpolated_points = linear_interpolation_half_segment(centroids_large[i], nearest_point)
                enhanced_points = np.vstack((enhanced_points, interpolated_points))

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(enhanced_points)

    return pcd


def point_idw(pcd, eps):
    points = np.asarray(pcd.points)
    labels = np.array(pcd.cluster_dbscan(eps, min_points=10, print_progress=False))
    clusters = {i: points[labels == i] for i in range(labels.max() + 1)}
    for i in clusters:
        for j in clusters:
            if i < j:
                dist_mat = distance_matrix(clusters[i], clusters[j])
                min_dist_idx = np.unravel_index(np.argmin(dist_mat, axis=None), dist_mat.shape)
                p1, p2 = clusters[i][min_dist_idx[0]], clusters[j][min_dist_idx[1]]
                pcd = connect_breaks_in_point_cloud(pcd, p1, p2)

    pcd = remove_noise_with_statistical_outlier(pcd, nb_neighbors=20, std_ratio=4.0)
    pcd = process_clusters(pcd)

    pcd = remove_noise_with_statistical_outlier(pcd, nb_neighbors=20, std_ratio=3.0)
    points = np.asarray(pcd.points)
    pcd = cluster_DBSCAN(pcd, points, eps1=0.03, eps2=0.05)
    pcd = process_clusters(pcd)

    points = np.asarray(pcd.points)
    pcd = cluster_DBSCAN(pcd, points, eps1=0.01, eps2=0.03)
    pcd = process_clusters(pcd)
    pcd = remove_noise_with_statistical_outlier(pcd, nb_neighbors=20, std_ratio=3.0)

    points = np.asarray(pcd.points)
    labels = np.array(pcd.cluster_dbscan(eps, min_points=10, print_progress=False))
    clusters = {i: points[labels == i] for i in range(labels.max() + 1)}
    for i in clusters:
        for j in clusters:
            if i < j:
                dist_mat = distance_matrix(clusters[i], clusters[j])
                min_dist_idx = np.unravel_index(np.argmin(dist_mat, axis=None), dist_mat.shape)
                p1, p2 = clusters[i][min_dist_idx[0]], clusters[j][min_dist_idx[1]]
                pcd = connect_breaks_in_point_cloud(pcd, p1, p2)

    # points = np.asarray(pcd.points)
    # with open(r"E:\Desktop\2222.txt", "w") as file:
    #     for point in points:
    #         line = ' '.join(map(str, point))
    #         file.write(line + '\n')

    o3d.visualization.draw_geometries([pcd], window_name="Connected Point Cloud")
    return pcd


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

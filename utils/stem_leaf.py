import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans


def split_stem_leaf(pcd, n_clusters=2):
    points = np.asarray(pcd.points)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
    labels = kmeans.labels_

    cluster1_indices = np.where(labels == 0)[0]
    cluster2_indices = np.where(labels == 1)[0]

    cluster1 = pcd.select_by_index(cluster1_indices)
    cluster2 = pcd.select_by_index(cluster2_indices)

    if len(cluster1.points) > len(cluster2.points):
        larger_cluster = cluster1
    else:
        larger_cluster = cluster2
    y_min = np.min(np.asarray(larger_cluster.points)[:, 1])
    return y_min


def extract_leaf_points(plant, y_min):
    points = np.asarray(plant.points)

    leaf_points = points[points[:, 1] > y_min]
    stem_points = points[points[:, 1] < y_min]

    leaf = o3d.geometry.PointCloud()
    leaf.points = o3d.utility.Vector3dVector(leaf_points)
    stem = o3d.geometry.PointCloud()
    stem.points = o3d.utility.Vector3dVector(stem_points)

    return leaf, stem

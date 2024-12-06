import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans


def txt_split(lines):
    output_zero = []
    output_one = []
    # lines = txt_ck(lines)  # x, y, z, nx, ny, nz, l1, l2
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 2:
            continue
        penultimate = parts[-2]
        if penultimate == '0.000000':
            new_line = ','.join(parts[:-2] + parts[-1:]) + '\n'
            output_zero.append(new_line)
        elif penultimate == '1.000000':
            new_line = ','.join(parts[:-2] + parts[-1:]) + '\n'
            output_one.append(new_line)
    return output_zero, output_one


def read_point(file):
    p = []
    for line in file:
        values = line.split(',')[:3]
        try:
            p.append([float(v) for v in values])
        except ValueError:
            print(f"Skipping invalid line: {line.strip()}")
            continue
    points_array = np.array(p)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    return pcd


def read_plant(file):
    points = []
    labels = []
    for line in file:
        values = line.split(',')
        if len(values) >= 4:
            point_values = [float(values[i]) for i in range(3)]
            label = float(values[-1])
            points.append(point_values)
            labels.append([label, label, label])

    points_array = np.array(points)
    labels_array = np.array(labels)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.colors = o3d.utility.Vector3dVector(labels_array)

    return pcd


def read_label(data):
    if data.shape[1] < 4:
        raise ValueError("Data format error: each row must contain at least 4 values (X, Y, Z, label)")

    points = data[:, :3]

    labels = data[:, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd, labels


def extract_points_and_labels(pcd):
    points = np.asarray(pcd.points)
    return points


def group_points_by_labels(points, labels, min_points=500, max_points=10000):
    label_to_points = {}
    for point, label in zip(points, labels):
        if label not in label_to_points:
            label_to_points[label] = []
        label_to_points[label].append(point)

    label_to_pcd = {}
    for label, points_list in label_to_points.items():
        points_array = np.array(points_list)
        num_points = len(points_list)
        if max_points >= num_points >= min_points:
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points_array)
            label_to_pcd[label] = new_pcd
        elif num_points > max_points:
            kmeans = KMeans(n_clusters=2, random_state=42)
            cluster_labels = kmeans.fit_predict(points_array)
            for i in range(2):
                cluster_points = points_array[cluster_labels == i]
                if len(cluster_points) >= min_points:
                    new_pcd = o3d.geometry.PointCloud()
                    new_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                    new_label = f"{label}_cluster_{i}"
                    label_to_pcd[new_label] = new_pcd

    return label_to_pcd

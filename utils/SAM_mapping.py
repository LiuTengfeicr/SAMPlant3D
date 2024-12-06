import os
import shutil
import open3d as o3d
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import mode
from point_sm import point_sm
from picture_noise import remove_pic_noise1
from point_noise import remove_noise
from matplotlib.patches import Rectangle


def map_masks_to_point_cloud(mask_dir, points_3d, scale_x, scale_z, offset_x, offset_z):
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    masks = [np.array(Image.open(os.path.join(mask_dir, f))) for f in mask_files]
    points_2d = points_3d[:, [0, 2]]
    # points_2d = points_3d[:, :2]

    tree = KDTree(points_2d)
    labels = np.zeros(len(points_3d), dtype=int)

    for idx, mask in enumerate(tqdm(masks, desc="Processing masks")):
        mask_y, mask_x = np.nonzero(mask)
        mask_points_2d = np.column_stack((((mask_x - offset_x) / scale_x), ((mask_y - offset_z) / scale_z)))
        distances, indices = tree.query(mask_points_2d, distance_upper_bound=0.05)
        valid_indices = indices[indices < len(labels)]
        labels[valid_indices] = idx + 1

        # visualize_points_and_masks(points_2d, mask_points_2d)

    labels = idw_interpolation(points_2d, labels, tree)
    labels = idw_interpolation(points_2d, labels, tree)

    points_3d_labeled = np.hstack((points_3d, labels.reshape(-1, 1)))
    return points_3d_labeled


def idw_interpolation(points_2d, labels, tree, power=2, k=20):
    unlabelled_indices = np.where(labels == 0)[0]
    for i in unlabelled_indices:
        distances, indices = tree.query(points_2d[i], k=k)
        distances = np.where(distances == 0, 1e-10, distances)
        mask = labels[indices] != 0
        valid_distances = distances[mask]
        valid_indices = indices[mask]
        valid_labels = labels[valid_indices]
        if len(valid_labels) > 0:
            weights = 1 / valid_distances ** power
            labels[i] = int(np.round(np.average(valid_labels, weights=weights)))
    return labels


def find_min_xz_points(points):
    min_x_index = np.argmin(points[:, 0])
    min_x_point = points[min_x_index]

    min_z_index = np.argmin(points[:, 2])
    min_z_point = points[min_z_index]

    return min_x_point, min_z_point


def find_max_xz_points(points):
    max_x_index = np.argmax(points[:, 0])
    max_x_point = points[max_x_index]

    max_z_index = np.argmax(points[:, 2])
    max_z_point = points[max_z_index]

    return max_x_point, max_z_point


def find_min_x_max_y_coordinates(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    black_pixels = np.where(image == 0)

    if black_pixels[0].size > 0 and black_pixels[1].size > 0:

        min_x_index = np.argmin(black_pixels[1])
        min_x_point = (black_pixels[1][min_x_index], black_pixels[0][min_x_index])

        max_y_index = np.argmax(black_pixels[0])
        max_y_point = (black_pixels[1][max_y_index], black_pixels[0][max_y_index])
    else:
        min_x_point, max_y_point = None, None

    return min_x_point, max_y_point


def find_max_x_min_y_coordinates(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None, None

    black_pixels = np.where(image == 0)

    if black_pixels[0].size == 0 or black_pixels[1].size == 0:
        print("No black pixels found in the image.")
        return None, None

    min_x_index = np.argmax(black_pixels[1])
    min_x_point = (black_pixels[1][min_x_index], black_pixels[0][min_x_index])

    max_y_index = np.argmin(black_pixels[0])
    max_y_point = (black_pixels[1][max_y_index], black_pixels[0][max_y_index])

    return min_x_point, max_y_point


def calculate_scale_and_offset(max_x_point, max_z_point, min_x_point, min_z_point, max_x_pic, min_y_pic, min_x_pic,
                               max_y_pic):
    x1, _ = max_x_point
    _, z1 = max_z_point

    x2, _ = min_x_point
    _, z2 = min_z_point

    a1, _ = max_x_pic
    _, b1 = min_y_pic

    a2, _ = min_x_pic
    _, b2 = max_y_pic

    scale_x = (a1 - a2) / (x1 - x2) if (x1 - x2) != 0 else float('inf')
    scale_z = (b1 - b2) / (z1 - z2) if (z1 - z2) != 0 else float('inf')

    offset_x = a1 - scale_x * x1
    offset_z = b1 - scale_z * z1

    return scale_x, scale_z, offset_x, offset_z


def min_x_min_z(points_3d, image_path):
    max_x_point, max_z_point = find_max_xz_points(points_3d)
    min_x_point, min_z_point = find_min_xz_points(points_3d)

    max_x_pic, min_y_pic = find_max_x_min_y_coordinates(image_path)
    min_x_pic, max_y_pic = find_min_x_max_y_coordinates(image_path)

    max_x_point = max_x_point[[0, 2]]
    max_z_point = max_z_point[[0, 2]]
    min_x_point = min_x_point[[0, 2]]
    min_z_point = min_z_point[[0, 2]]

    scale_x, scale_z, offset_x, offset_z = calculate_scale_and_offset(max_x_point, max_z_point, min_x_point,
                                                                      min_z_point, max_x_pic, min_y_pic, min_x_pic,
                                                                      max_y_pic)

    return scale_x, scale_z, offset_x, offset_z


def process_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    if width != 1000 or height != 1000:
        raise ValueError("Images must be 1000x1000 pixels")
    white_img = Image.new("RGB", (1000, 1000), "white")

    left = (width - 800) // 2
    top = (height - 800) // 2
    right = left + 800
    bottom = top + 800

    center_crop = img.crop((left, top, right, bottom))
    white_img.paste(center_crop, (left, top))

    white_img.save(image_path)


def remove_noise_with_statistical_outlier(pcd, nb_neighbors=50, std_ratio=2.0):
    clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    all_indices = set(range(len(pcd.points)))

    outlier_indices = all_indices - set(ind)

    outlier_pcd = pcd.select_by_index(list(outlier_indices))
    return clean_pcd, outlier_pcd


def splicing_point(points_3d_labeled, outlier_points):
    positions = points_3d_labeled[:, :3]
    labels = points_3d_labeled[:, 3]

    tree = KDTree(positions)

    distances, indices = tree.query(outlier_points, k=3)

    nearest_labels = labels[indices]

    assigned_labels, _ = mode(nearest_labels, axis=1)

    outlier_points_labeled = np.hstack((outlier_points, assigned_labels.reshape(-1, 1)))

    combined_points = np.vstack((points_3d_labeled, outlier_points_labeled))
    return combined_points


def SAMpoint(args, plant_point, base_filename):
    mask_dir = os.path.join(args.sam_mask, base_filename)
    for filename1 in os.listdir(mask_dir):
        if filename1.endswith('.png'):
            remove_pic_noise1(os.path.join(mask_dir, filename1), method='median')
    target_path = os.path.join(args.sam_mask, base_filename, '0', '0.png')

    process_image(target_path)

    # _, points_3d, _ = remove_noise(False, plant_point, False)
    points_3d, outlier_pcd = remove_noise_with_statistical_outlier(plant_point, nb_neighbors=50, std_ratio=2.0)

    points_3d = np.asarray(points_3d.points)
    outlier_points = np.asarray(outlier_pcd.points)

    scale_x, scale_z, offset_x, offset_z = min_x_min_z(points_3d, target_path)

    points_3d_labeled = map_masks_to_point_cloud(mask_dir, points_3d, scale_x, scale_z, offset_x, offset_z)
    points_3d_labeled = splicing_point(points_3d_labeled, outlier_points)
    points_3d_labeled = point_sm(points_3d_labeled)
    # np.savetxt(args.output_pointcloud + '/' + f'{base_filename}.txt', points_3d_labeled, fmt="%f %f %f %d")
    return points_3d_labeled

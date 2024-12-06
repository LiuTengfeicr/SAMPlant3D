import json
import os
import numpy as np
import utils.utils as SAMPoint
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def save_txt_point_cloud(file_path, pcd):
    points = np.asarray(pcd.points)
    np.savetxt(file_path, points, fmt='%f')


def save_point_cloud_with_labels(pcd, output_file_path):
    points = np.asarray(pcd.points)
    labels = np.asarray(pcd.colors)[:, 0]

    with open(output_file_path, 'w') as file:
        for point, label in zip(points, labels):
            line = f"{point[0]} {point[1]} {point[2]} {label}\n"
            file.write(line)


def main(args, filename):
    global elapsed_time, piou, precision, recall, f1
    with open(args.file_path + '/' + filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    tray, plant_point = SAMPoint.txt_split(lines)
    if plant_point:
        tray = SAMPoint.read_point(tray)
        plant_point = SAMPoint.read_plant(plant_point)

        tray, _, plant_point = SAMPoint.remove_noise(tray, False, plant_point)

        tray, rotation_matrix = SAMPoint.rotate_point(tray)
        _, scale = SAMPoint.plate_scale(tray)

        plant_point.rotate(rotation_matrix)

        base_filename = os.path.splitext(filename)[0]
        pic_path = SAMPoint.projection_xy(plant_point, args.input_pic, base_filename)

        _, pic_path = SAMPoint.remove_pic_noise(pic_path, method='median')

        start_time = time.time()

        args.sam_pic = pic_path
        args.sam_mask = args.output_mask
        SAMPoint.sam(args)

        points_3d_labeled = SAMPoint.SAMpoint(args, plant_point, base_filename)

        end_time = time.time()
        elapsed_time = end_time - start_time
        piou, precision, recall, f1 = SAMPoint.calculate_metrics(plant_point, points_3d_labeled)

        if args.measure:
            plant_point, labels = SAMPoint.read_label(points_3d_labeled)
            points = SAMPoint.extract_points_and_labels(plant_point)
            grouped_pcds = SAMPoint.group_points_by_labels(points, labels, args.min_points)
            plant_heights = {}
            plant_Leaf_Angles = {}
            plant_leaf_areas = {}
            plant_Leaf_Lengths = {}
            plant_Volumes = {}
            plant_Leaf_WidthS = {}
            for label, plant in tqdm(grouped_pcds.items(), desc=f"Processing Plants {filename}"):
                _, plant, _ = SAMPoint.remove_noise(False, plant, False)

                leaf_y = SAMPoint.split_stem_leaf(plant)
                try:
                    plant = SAMPoint.point_idw(plant, args.eps)
                except ValueError as e:
                    print(f"Skipping interpolation for {label} due to error: {e}")
                if leaf_y:
                    leaf, stem = SAMPoint.extract_leaf_points(plant, leaf_y)
                    plant_height, plant_Leaf_Width, plant_Leaf_Length, plant_leaf_area, plant_Alpha_Volume, plant_Normalized_Volume, plant_Leaf_Angle = SAMPoint.plant_measure_leaf(
                        plant, scale, args, leaf, stem)
                else:
                    plant_height, plant_Leaf_Width, plant_Leaf_Length, plant_leaf_area, plant_Alpha_Volume, plant_Normalized_Volume, plant_Leaf_Angle = SAMPoint.plant_measure_math(
                        plant, scale, args)

                plant_heights[label] = plant_height
                plant_Leaf_WidthS[label] = plant_Leaf_Width
                plant_Leaf_Angles[label] = plant_Leaf_Angle
                plant_leaf_areas[label] = plant_leaf_area
                plant_Leaf_Lengths[label] = plant_Leaf_Length
                if args.Normalized_Volume:
                    plant_Volumes[label] = plant_Normalized_Volume
                else:
                    plant_Volumes[label] = plant_Alpha_Volume
            results = {
                'plant_height': plant_heights,
                'plant_leaf_area': plant_leaf_areas,
                'plant_Volume': plant_Volumes,
                'plant_Leaf_Width': plant_Leaf_WidthS,
                'plant_Leaf_Length': plant_Leaf_Lengths,
                'plant_Leaf_Angle': plant_Leaf_Angles,
                'piou': piou,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'elapsed_time': elapsed_time
            }
        else:
            results = {
                'piou': piou,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'elapsed_time': elapsed_time
            }
        results_path = os.path.join(args.output_measure, f'{base_filename}.json')
        with open(results_path, mode='w') as f:
            json.dump(results, f, indent=4)
    return piou, elapsed_time, precision, recall, f1


def start(args):
    piou_sum = 0
    time_sum = 0
    count = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    for filename in os.listdir(args.file_path):
        if filename.endswith('.txt'):
            try:
                piou, elapsed_time, p_precision, p_recall, p_f1 = main(args, filename)
            except ValueError as e:
                print(f"Error processing {filename}: {e}")
                continue
            piou_sum += piou
            time_sum += elapsed_time
            precision_sum += p_precision
            recall_sum += p_recall
            f1_sum += p_f1
            count += 1
    siou = piou_sum / count if count > 0 else 0
    precision = precision_sum / count if count > 0 else 0
    recall = recall_sum / count if count > 0 else 0
    f1 = f1_sum / count if count > 0 else 0
    results = {
        'siou': siou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'time_sum': time_sum,
    }
    results_path = os.path.join(args.output_measure, f'metrics.json')
    with open(results_path, mode='w') as f:
        json.dump(results, f, indent=4)

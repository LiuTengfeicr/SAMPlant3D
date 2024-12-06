import numpy as np
from sklearn.neighbors import NearestNeighbors


def extract_labels_from_colors(pcd):
    return np.asarray(pcd.colors)[:, 0].astype(int)


def match_points(source_data, target_data):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(source_data[:, :3])
    distances, indices = neigh.kneighbors(target_data[:, :3])
    return distances.flatten(), indices.flatten()


def calculate_metrics_for_label(before_data, after_data, matched_indices):
    before_labels = before_data[:, -1].astype(int)
    after_labels = after_data[:, -1].astype(int)

    unique_labels = np.unique(after_labels)
    iou_list, precision_list, recall_list, f1_list = [], [], [], []

    for label in unique_labels:
        label_mask = (after_labels == label)
        matched_labels_in_before = before_labels[matched_indices[label_mask]]

        if len(matched_labels_in_before) > 0:
            most_common_label = np.bincount(matched_labels_in_before).argmax()

            max_label_mask = (matched_labels_in_before == most_common_label)
            TP = np.sum(max_label_mask)
            FN = np.sum(before_labels == most_common_label) - TP
            FP = len(matched_labels_in_before) - TP

            iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            iou_list.append(iou)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

    mean_iou = np.mean(iou_list) if iou_list else 0
    mean_precision = np.mean(precision_list) if precision_list else 0
    mean_recall = np.mean(recall_list) if recall_list else 0
    mean_f1 = np.mean(f1_list) if f1_list else 0

    return mean_iou, mean_precision, mean_recall, mean_f1


def calculate_metrics(before_point, after_points):
    before_point_labels = extract_labels_from_colors(before_point)
    before_point_data = np.hstack((np.asarray(before_point.points), before_point_labels.reshape(-1, 1)))
    _, matched_indices = match_points(before_point_data, after_points)
    miou, precision, recall, f1 = calculate_metrics_for_label(before_point_data, after_points, matched_indices)
    return miou, precision, recall, f1

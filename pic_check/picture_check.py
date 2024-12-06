import cv2
import numpy as np
import math
import os
import json

from sklearn.cluster import KMeans
from tqdm import tqdm


# brenner
def brenner(img):
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 2):
        for y in range(0, shape[1]):
            out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
    return out


# Laplacian
def Laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


# SMD
def SMD(img):
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0] - 1):
        for y in range(0, shape[1]):
            out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
            out += math.fabs(int(img[x, y] - int(img[x + 1, y])))
    return out


def SMD2(img):
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            out += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(int(img[x, y] - int(img[x, y + 1])))
    return out


def variance(img):
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            out += (img[x, y] - u) ** 2
    return out


def energy(img):
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            out += ((int(img[x + 1, y]) - int(img[x, y])) ** 2) * ((int(img[x, y + 1] - int(img[x, y]))) ** 2)
    return out


def Vollath(img):
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0] * shape[1] * (u ** 2)
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1]):
            out += int(img[x, y]) * int(img[x + 1, y])
    return out


def entropy(img):
    out = 0
    count = np.shape(img)[0] * np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i] != 0:
            out -= p[i] * math.log(p[i] / count) / count
    return out


def main(folder_path, save_path):
    results = {
        "brenner": [],
        "Laplacian": [],
        "SMD": [],
        "SMD2": [],
        "variance": [],
        "energy": [],
        "Vollath": [],
        "entropy": []
    }

    for filename in tqdm(os.listdir(folder_path), desc="Processing images"):
        if filename.endswith('.jpg'):
            full_path = os.path.join(folder_path, filename)
            image = cv2.imread(full_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            results["brenner"].append({"filename": filename, "score": brenner(gray_image)})
            results["Laplacian"].append({"filename": filename, "score": Laplacian(gray_image)})
            results["SMD"].append({"filename": filename, "score": SMD(gray_image)})
            results["SMD2"].append({"filename": filename, "score": SMD2(gray_image)})
            results["variance"].append({"filename": filename, "score": variance(gray_image)})
            results["energy"].append({"filename": filename, "score": energy(gray_image)})
            results["Vollath"].append({"filename": filename, "score": Vollath(gray_image)})
            results["entropy"].append({"filename": filename, "score": entropy(gray_image)})

    for method in results:
        results[method] = sorted(results[method], key=lambda x: x["score"], reverse=True)

    with open('sorted_results_per_method.json', 'w') as file:
        json.dump(results, file, indent=4)


def guiyi(input_list):
    # |i-min|/|max-min|
    result = []
    input_max = max(input_list)
    input_min = min(input_list)
    mi = input_max - input_min
    if mi == 0:
        return [0] * len(input_list)
    for i in input_list:
        result.append(abs(i - input_min) / mi)
    return result


def write():
    with open('sorted_results_per_method.json', 'r') as file:
        results = json.load(file)

    anomalies_results = {}

    for method, data_list in results.items():

        data = np.array([item['score'] for item in data_list])
        data_sorted = np.sort(data)

        slopes = np.diff(data_sorted) / 0.1  # y2-y1 / x2-x1

        if len(slopes) > 4:
            min_slopes = np.partition(slopes, 3)[:4]
            threshold = np.max(min_slopes)
        else:
            threshold = np.max(slopes) if slopes.size > 0 else 0

        significant_changes = np.where(slopes > 5 * threshold)[0]

        if significant_changes.size > 0:
            first_significant_change = significant_changes[0]
            anomalies = np.arange(first_significant_change + 1, len(data_sorted))
        else:
            anomalies = []

        anomalies_results[method] = {
            "anomalies_filenames": anomalies.tolist()
        }

    with open('anomalies_results.json', 'w') as file:
        json.dump(anomalies_results, file, indent=4)


if __name__ == '__main__':
    # folder_path = r'E:\Desktop\test'
    # save_path = r'E:\Desktop\test\results.json'
    # main(folder_path, save_path)
    write()

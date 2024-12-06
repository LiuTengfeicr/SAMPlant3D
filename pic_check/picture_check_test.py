import json
import numpy as np
import matplotlib.pyplot as plt


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


def median_of_medians(data, median):
    sorted_data = sorted(data)

    n = len(sorted_data)

    left_data = sorted_data[:n // 2]
    left_median = (left_data[len(left_data) // 2 - 1] + left_data[len(left_data) // 2]) / 2 if len(
        left_data) % 2 == 0 else left_data[len(left_data) // 2]

    right_data = sorted_data[(n + 1) // 2:]
    right_median = (right_data[len(right_data) // 2 - 1] + right_data[len(right_data) // 2]) / 2 if len(
        right_data) % 2 == 0 else right_data[len(right_data) // 2]

    median_of_medians = (left_median + right_median + median) / 2

    return median_of_medians


if __name__ == '__main__':
    with open('sorted_results_per_method.json', 'r') as file:
        results = json.load(file)

    plt.figure(figsize=(10, 6))

    for method, data_list in results.items():
        data = np.array([item['score'] for item in data_list])
        data = guiyi(data)
        data_sorted = np.sort(data)

        slopes = np.diff(data_sorted)
        threshold = np.median(slopes)
        # threshold = median_of_medians(slopes, threshold)

        significant_changes = np.where(slopes > 10 * threshold)[0]
        anomalies = np.arange(significant_changes[0] + 1, len(data_sorted)) if significant_changes.size > 0 else []

        plt.plot(data_sorted, label=f'Data ({method})')
        plt.scatter(anomalies, data_sorted[anomalies], color='red', label=f'Anomalies ({method})', s=50)

    plt.title("Data Points and Detected Anomalies by Method")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

import math
from plant_measure import *


def plant_measure_math(plant_point, scale, args):
    plant_height, plant_Leaf_Width, plant_Leaf_Length = hight(plant_point, scale)
    plant_Alpha_Volume, plant_Normalized_Volume = None, None
    plant_leaf_area = compute_surface_area(plant_point, args.alpha_s) * scale**2 - (2*plant_height/3) * 0.02 * scale * math.pi
    _, _, y_Angle, plant_Leaf_Angle, _ = calculate_vectors_angle(plant_point)
    if args.Normalized_Volume:
        plant_Normalized_Volume = calculate_normalized_volume(plant_point, normalization_factor=1000.0) * scale**3
    else:
        plant_Alpha_Volume = compute_alpha_shape_volume(plant_point, args.alpha_v) * scale**3

    y_Angle_rad = math.radians(y_Angle)
    return plant_height, plant_Leaf_Width, plant_Leaf_Length/math.cos(y_Angle_rad), plant_leaf_area/2, plant_Alpha_Volume, plant_Normalized_Volume, plant_Leaf_Angle


def plant_measure_leaf(plant_point, scale, args, leaf, stem):
    plant_height, plant_Leaf_Width, plant_Leaf_Length = hight(plant_point, scale)
    plant_Alpha_Volume, plant_Normalized_Volume = None, None
    plant_leaf_area = compute_surface_area(leaf, args.alpha_s) * scale**2
    _, _, y_Angle, plant_Leaf_Angle = calculate_leaf_angle(leaf, stem)
    if args.Normalized_Volume:
        plant_Normalized_Volume = calculate_normalized_volume(plant_point, normalization_factor=1000.0) * scale**3
    else:
        plant_Alpha_Volume = compute_alpha_shape_volume(plant_point, args.alpha_v) * scale**3

    y_Angle_rad = math.radians(y_Angle)
    return plant_height, plant_Leaf_Width, plant_Leaf_Length/math.cos(y_Angle_rad), plant_leaf_area/2, plant_Alpha_Volume, plant_Normalized_Volume, plant_Leaf_Angle

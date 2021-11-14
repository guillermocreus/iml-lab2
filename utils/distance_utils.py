import numpy as np


def compute_l1_distance(v1, v2):
    return (abs(np.array(v1) - np.array(v2))).sum()


def compute_l2_distance(v1, v2):
    return np.sqrt(((np.array(v1) - np.array(v2)) ** 2).sum())


# variable for getting distance implementations by name
distance_dict = {
    'l1': compute_l1_distance,
    'l2': compute_l2_distance,
    'manhattan': compute_l1_distance
}

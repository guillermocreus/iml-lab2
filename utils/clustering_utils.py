from math import inf
import numpy as np


# TODO add documentation

def assign_cluster(x, clusters_coord, distance_metric):
	"""

	"""
	cluster, distance = -1, inf
	for i, c_i in enumerate(clusters_coord):
		aux_dist = distance_metric(x, c_i)
		if aux_dist < distance:
			cluster = i
			distance = aux_dist
	return cluster, distance


def change_in_clusters(v, v_old, eps=0.01):
	"""

	"""
	diff = np.abs(v - v_old).sum(axis=1).max()
	if diff > eps:
		return True
	return False


def compute_labels(df, v, distance_metric):
	"""

	"""
	labels = np.zeros(df.shape[0])
	for i, x in enumerate(df):
		cluster, _ = assign_cluster(x, v, distance_metric)
		labels[i] = cluster
	return labels

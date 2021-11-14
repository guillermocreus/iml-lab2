import numpy as np
import pandas as pd
import umap

import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import calinski_harabasz_score
# from sklearn.metrics import davies_bouldin_score
# from sklearn.decomposition import PCA
# from classifiers.fuzzy_c_means import FuzzyCMeans
# from classifiers.k_means import KMeans
# from classifiers.k_medF import KMedoidFast
# from classifiers.NBOptics import Optics
# from utils.distance_utils import distance_dict
# import matplotlib.colors as colors
# from datetime import datetime
#
# colors = list(colors._colors_full_map.values())
from pandas import Series, DataFrame

from pca import PCA
from utils.ploting_utils import plot_scatter


class Aggregator:
	"""
		Class which will aggregate the results from k Means and UMAP
	"""

	def __init__(self):
		self._transformed_data = {'PCA': {}, 'umap': {}}
		self._results = {'PCA': {}, 'umap': {}}
		self._metrics = {'PCA': {}, 'umap': {}}
		self._final_metrics = {'PCA': {}, 'umap': {}}
		self._confusion_matrix = dict()

	def fit_UMAP(self, data, y: Series, dataset_name='', plot=True):
		"""

		Parameters
		----------
		data
		y -> pass it as Series # y['class']
		dataset_name
		plot

		Returns
		-------

		"""
		fit = umap.UMAP()
		u = fit.fit_transform(data)

		if plot:
			plot_scatter(u, y, f'UMAP embedding of {dataset_name} dataset')

		self._transformed_data['umap'] = u

	def fit_PCA(self, data, desired_components, y, dataset_name='', plot=True):

		pca = PCA()
		results = pca.fit_transform(data, desired_components)
		if plot:
			plot_scatter(results['reduced'], y, f'reduce data {dataset_name}')

		return results['reduced']

	def evaluate(self, data, y, dataset_name=''):
		# km = KMeans()
		return 0
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.decomposition import IncrementalPCA
from classifiers.k_means import KMeans
from utils.distance_utils import distance_dict
import time
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
		self._transformed_data = {'pca': {}, 'umap': {}, 'pca_sklearn': {}, 'incremental_pca': {}}
		self._results = {'complete': {}, 'pca': {}, 'umap': {}, 'pca_sklearn': {}, 'incremental_pca': {}}
		self._metrics = {'complete': {}, 'pca': {}, 'umap': {}, 'pca_sklearn': {}, 'incremental_pca': {}}
		self._confusion_matrix = dict()

	def fit_UMAP(self, data, y: Series, umap_parameters, dataset_name='', plot=True):
		fit = umap.UMAP(
			n_neighbors=umap_parameters['n_neighbors'],
			min_dist=umap_parameters['min_dist'],
			metric=umap_parameters['metric']
		)
		u = fit.fit_transform(data)

		if plot:
			plot_scatter(u, y, f'UMAP embedding of {dataset_name} dataset')

		return u

	def fit_PCA(self, data, desired_components, y, dataset_name='', plot=True):

		pca = PCA()
		results = pca.fit_transform(data, desired_components)
		if plot:
			plot_scatter(results['reduced'], y, f'PCA reduced data of dataset {dataset_name}')

		return results

	def fit_PCA_sklearn(self, data, desired_components, y, dataset_name='', plot=True):

		pca = PCA_sklearn(n_components=desired_components)
		results = pca.fit_transform(data)
		if plot:
			plot_scatter(results, y, f'Sklearn PCA reduced data of dataset {dataset_name}')

		return results

	def fit_incremental_PCA(self, data, desired_components, y, dataset_name='', plot=True):

		pca = IncrementalPCA(n_components=desired_components, batch_size=10)
		results = pca.fit_transform(data)
		if plot:
			plot_scatter(results, y, f'Sklearn Incremental PCA reduced data of dataset {dataset_name}')
		return results

	def evaluate(self, data, desired_components, y, umap_parameters, dataset_name=''):

		results_umap = self.fit_UMAP(data, y, umap_parameters, dataset_name=dataset_name)
		self._transformed_data['umap'] = results_umap

		results_pca = self.fit_PCA(data, desired_components, y, dataset_name=dataset_name)
		self._transformed_data['pca'] = results_pca['reduced']

		results_pca_sklearn = self.fit_PCA_sklearn(data, desired_components,
												   y, dataset_name=dataset_name)
		self._transformed_data['pca_sklearn'] = results_pca_sklearn

		results_incremental_pca_sklearn = self.fit_incremental_PCA(data, desired_components,
																   y, dataset_name=dataset_name)
		self._transformed_data['incremental_pca'] = results_incremental_pca_sklearn

		km = KMeans()
		completeDatatime={"total":[],"predict":[],"train":[]}
		PCADatatime={"total":[],"predict":[],"train":[]}
		SkPCADatatime = {"total": [], "predict": [], "train": []}
		UMAPDatatime={"total":[],"predict":[],"train":[]}
		IncPCADatatime={"total":[],"predict":[],"train":[]}
		for n_cluster in range(2, 10):
			# complete data
			start = time.perf_counter()
			_ = km.train(data, n_cluster, distance_dict['l2'])
			mid = time.perf_counter()
			labels_complete = km.predict(data, distance_dict['l2'])
			end = time.perf_counter()
			completeDatatime["train"].append(mid - start)
			completeDatatime["predict"].append(end - mid)
			completeDatatime["total"].append(end - start)

			self._results['complete'][n_cluster] = labels_complete

			sh_score_complete = silhouette_score(data, labels_complete)
			calinski_harabasz_score_complete = calinski_harabasz_score(data, labels_complete)
			davies_bouldin_score_complete = davies_bouldin_score(data, labels_complete)

			self._metrics['complete'][n_cluster] = {
				'silhouette_score': sh_score_complete,
				'calinski_harabasz_score': calinski_harabasz_score_complete,
				'davies_bouldin_score': davies_bouldin_score_complete
			}

			# homemade PCA reduced data
			start = time.perf_counter()
			_ = km.train(self._transformed_data['pca'], n_cluster, distance_dict['l2'])
			mid = time.perf_counter()
			labels_pca = km.predict(self._transformed_data['pca'], distance_dict['l2'])
			end = time.perf_counter()
			PCADatatime["train"].append(mid - start)
			PCADatatime["predict"].append(end - mid)
			PCADatatime["total"].append(end - start)

			self._results['pca'][n_cluster] = labels_pca

			sh_score_pca = silhouette_score(self._transformed_data['pca'], labels_pca)
			calinski_harabasz_score_pca = calinski_harabasz_score(self._transformed_data['pca'], labels_pca)
			davies_bouldin_score_pca = davies_bouldin_score(self._transformed_data['pca'], labels_pca)

			self._metrics['pca'][n_cluster] = {
				'silhouette_score': sh_score_pca,
				'calinski_harabasz_score': calinski_harabasz_score_pca,
				'davies_bouldin_score': davies_bouldin_score_pca
			}

			# sklearn PCA reduced data
			start = time.perf_counter()
			_ = km.train(self._transformed_data['pca_sklearn'], n_cluster, distance_dict['l2'])
			mid = time.perf_counter()
			labels_pca_sklearn = km.predict(self._transformed_data['pca_sklearn'], distance_dict['l2'])
			end = time.perf_counter()
			SkPCADatatime["train"].append(mid - start)
			SkPCADatatime["predict"].append(end - mid)
			SkPCADatatime["total"].append(end - start)

			self._results['pca_sklearn'][n_cluster] = labels_pca_sklearn



			sh_score_pca_sklearn = silhouette_score(self._transformed_data['pca_sklearn'], labels_pca_sklearn)
			calinski_harabasz_score_pca_sklearn = calinski_harabasz_score(self._transformed_data['pca_sklearn'], labels_pca_sklearn)
			davies_bouldin_score_pca_sklearn = davies_bouldin_score(self._transformed_data['pca_sklearn'], labels_pca_sklearn)

			self._metrics['pca_sklearn'][n_cluster] = {
				'silhouette_score': sh_score_pca_sklearn,
				'calinski_harabasz_score': calinski_harabasz_score_pca_sklearn,
				'davies_bouldin_score': davies_bouldin_score_pca_sklearn
			}

			# umap reduced data
			start = time.perf_counter()
			_ = km.train(self._transformed_data['umap'], n_cluster, distance_dict['l2'])
			mid = time.perf_counter()
			labels_umap = km.predict(self._transformed_data['umap'], distance_dict['l2'])
			end = time.perf_counter()
			UMAPDatatime["train"].append(mid - start)
			UMAPDatatime["predict"].append(end - mid)
			UMAPDatatime["total"].append(end - start)

			self._results['umap'][n_cluster] = labels_umap



			sh_score_umap = silhouette_score(self._transformed_data['umap'], labels_umap)
			calinski_harabasz_score_umap = calinski_harabasz_score(self._transformed_data['umap'], labels_umap)
			davies_bouldin_score_umap = davies_bouldin_score(self._transformed_data['umap'], labels_umap)

			self._metrics['umap'][n_cluster] = {
				'silhouette_score': sh_score_umap,
				'calinski_harabasz_score': calinski_harabasz_score_umap,
				'davies_bouldin_score': davies_bouldin_score_umap
			}

			# incremental pca

			start = time.perf_counter()
			_ = km.train(self._transformed_data['incremental_pca'], n_cluster, distance_dict['l2'])
			mid = time.perf_counter()
			labels_incremental_pca_sklearn = km.predict(self._transformed_data['incremental_pca'], distance_dict['l2'])
			end = time.perf_counter()
			IncPCADatatime["train"].append(mid - start)
			IncPCADatatime["predict"].append(end - mid)
			IncPCADatatime["total"].append(end - start)

			self._results['incremental_pca'][n_cluster] = labels_pca_sklearn

			sh_score_incremental_pca_sklearn = silhouette_score(self._transformed_data['incremental_pca'], labels_incremental_pca_sklearn)
			calinski_harabasz_score_incremental_pca_sklearn = calinski_harabasz_score(self._transformed_data['incremental_pca'], labels_incremental_pca_sklearn)
			davies_bouldin_score_incremental_pca_sklearn = davies_bouldin_score(self._transformed_data['incremental_pca'], labels_incremental_pca_sklearn)

			self._metrics['incremental_pca'][n_cluster] = {
				'silhouette_score': sh_score_incremental_pca_sklearn,
				'calinski_harabasz_score': calinski_harabasz_score_incremental_pca_sklearn,
				'davies_bouldin_score': davies_bouldin_score_incremental_pca_sklearn
			}


		print("Average Complete Train Time:"+ str(sum(completeDatatime["train"])/len(completeDatatime["train"])))
		print("Average Complete Predict Time:"+ str(sum(completeDatatime["predict"])/len(completeDatatime["predict"])))
		print("Average Complete Total Time:"+ str(sum(completeDatatime["total"])/len(completeDatatime["total"])))

		print("Average PCA developed Train Time:"+ str(sum(PCADatatime["train"])/len(PCADatatime["train"])))
		print("Average PCA developed Predict Time:"+ str(sum(PCADatatime["predict"])/len(PCADatatime["predict"])))
		print("Average PCA developed Total Time:"+ str(sum(PCADatatime["total"])/len(PCADatatime["total"])))

		print("Average PCA sklearn Train Time:"+ str(sum(SkPCADatatime["train"])/len(SkPCADatatime["train"])))
		print("Average PCA sklearn Predict Time:"+ str(sum(SkPCADatatime["predict"])/len(SkPCADatatime["predict"])))
		print("Average PCA sklearn Total Time:"+ str(sum(SkPCADatatime["total"])/len(SkPCADatatime["total"])))

		print("Average UMAP Train Time:"+ str(sum(UMAPDatatime["train"])/len(UMAPDatatime["train"])))
		print("Average UMAP Predict Time:"+ str(sum(UMAPDatatime["predict"])/len(UMAPDatatime["predict"])))
		print("Average UMAP Total Time:"+ str(sum(UMAPDatatime["total"])/len(UMAPDatatime["total"])))

		print("Average PCA increment Train Time:"+ str(sum(IncPCADatatime["train"])/len(IncPCADatatime["train"])))
		print("Average PCA increment Predict Time:"+ str(sum(IncPCADatatime["predict"])/len(IncPCADatatime["predict"])))
		print("Average PCA increment Total Time:"+ str(sum(IncPCADatatime["total"])/len(IncPCADatatime["total"])))
		return 0


	def plot_metrics_with_error(self):
		"""
			plot error metrics
		"""
		_colors = {
			'silhouette_score': 'b',
			'calinski_harabasz_score': 'g',
			'davies_bouldin_score': 'y'
		}

		fig, axs = plt.subplots(nrows=3, ncols=len(self._metrics.keys()), figsize=(30, 15))

		for ax, col in zip(axs[0, :], self._metrics.keys()):
			ax.set_title(col, size=14)

		for ax, row in zip(axs[:, 0], ['Silhouette', 'Calinski', 'Davies Bouldin']):
			ax.set_ylabel(row, size=14)

		for i, (method, cluster_dict) in enumerate(self._metrics.items()):

			scores_method = {
				'silhouette_score': [],
				'calinski_harabasz_score': [],
				'davies_bouldin_score': []
			}

			for (n_cluster, scores_dict) in cluster_dict.items():
				for key, value in scores_dict.items():
					scores_method[key].append(value)

			for j, (score, results) in enumerate(scores_method.items()):
				axs[j][i].plot(np.arange(2, 10), results, marker='o', color=_colors[score])
		plt.show()

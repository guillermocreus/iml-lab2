import numpy
import numpy as np
from pandas import Series, DataFrame
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

		# External metrics
		self._precision = {'complete': {}, 'pca': {}, 'umap': {}, 'pca_sklearn': {}, 'incremental_pca': {}}
		self._recall = {'complete': {}, 'pca': {}, 'umap': {}, 'pca_sklearn': {}, 'incremental_pca': {}}
		self._f1_score = {'complete': {}, 'pca': {}, 'umap': {}, 'pca_sklearn': {}, 'incremental_pca': {}}

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

		completeDatatime={"total":[],"cluster":np.array([]),"alg":np.zeros(len(range(2, 10)))}
		PCADatatime={"total": [],"cluster":np.array([]),"alg":[]}
		SkPCADatatime = {"total": [], "cluster": np.array([]), "alg": []}
		UMAPDatatime={"total": [],"cluster":np.array([]),"alg": []}
		IncPCADatatime={"total": [],"cluster":np.array([]),"alg": []}

		start = time.perf_counter()
		results_umap = self.fit_UMAP(data, y, umap_parameters, dataset_name=dataset_name)
		total=time.perf_counter()-start
		UMAPDatatime["alg"]=np.ones(len(range(2,10)))*total
		self._transformed_data['umap'] = results_umap

		start = time.perf_counter()
		results_pca = self.fit_PCA(data, desired_components, y, dataset_name=dataset_name)
		total = time.perf_counter() - start
		PCADatatime["alg"] = np.ones(len(range(2, 10))) * total
		print("\nPCA data:")
		print(results_pca['reduced'])

		self._transformed_data['pca'] = results_pca['reduced']

		start = time.perf_counter()
		results_pca_sklearn = self.fit_PCA_sklearn(data, desired_components,
												   y, dataset_name=dataset_name)
		total = time.perf_counter() - start
		SkPCADatatime["alg"] = np.ones(len(range(2, 10))) * total
		self._transformed_data['pca_sklearn'] = results_pca_sklearn
		print("\nSklearn PCA data:")
		print(results_pca_sklearn)

		start = time.perf_counter()
		results_incremental_pca_sklearn = self.fit_incremental_PCA(data, desired_components,
																   y, dataset_name=dataset_name)
		total = time.perf_counter() - start
		IncPCADatatime["alg"] = np.ones(len(range(2, 10))) * total

		print("\nIncremental PCA data:")
		print(results_incremental_pca_sklearn)

		self._transformed_data['incremental_pca'] = results_incremental_pca_sklearn

		km = KMeans()

		for n_cluster in range(2, 10):
			# complete data
			start = time.perf_counter()
			_ = km.train(data, n_cluster, distance_dict['l2'])
			labels_complete = km.predict(data, distance_dict['l2'])
			end = time.perf_counter()
			completeDatatime["cluster"]=np.append(completeDatatime["cluster"],(end - start))

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
			labels_pca = km.predict(self._transformed_data['pca'], distance_dict['l2'])
			end = time.perf_counter()
			PCADatatime["cluster"]=np.append(PCADatatime["cluster"],(end - start))

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
			labels_pca_sklearn = km.predict(self._transformed_data['pca_sklearn'], distance_dict['l2'])
			end = time.perf_counter()
			SkPCADatatime["cluster"]=np.append(SkPCADatatime["cluster"],(end - start))

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
			labels_umap = km.predict(self._transformed_data['umap'], distance_dict['l2'])
			end = time.perf_counter()
			UMAPDatatime["cluster"]=np.append(UMAPDatatime["cluster"],(end - start))

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
			labels_incremental_pca_sklearn = km.predict(self._transformed_data['incremental_pca'], distance_dict['l2'])
			end = time.perf_counter()
			IncPCADatatime["cluster"]=np.append(IncPCADatatime["cluster"],(end - start))

			self._results['incremental_pca'][n_cluster] = labels_pca_sklearn

			sh_score_incremental_pca_sklearn = silhouette_score(self._transformed_data['incremental_pca'], labels_incremental_pca_sklearn)
			calinski_harabasz_score_incremental_pca_sklearn = calinski_harabasz_score(self._transformed_data['incremental_pca'], labels_incremental_pca_sklearn)
			davies_bouldin_score_incremental_pca_sklearn = davies_bouldin_score(self._transformed_data['incremental_pca'], labels_incremental_pca_sklearn)

			self._metrics['incremental_pca'][n_cluster] = {
				'silhouette_score': sh_score_incremental_pca_sklearn,
				'calinski_harabasz_score': calinski_harabasz_score_incremental_pca_sklearn,
				'davies_bouldin_score': davies_bouldin_score_incremental_pca_sklearn
			}

	def compute_confusion_matrix(self, n_cluster_dict, true_labels):
		"""
		"""
		for algo in list(self._results):
			pred_labels = self._results[algo][n_cluster_dict[algo]]
			TP, FP, TN, FN = 0, 0, 0, 0
			for i in range(len(pred_labels)):
				for j in range(i + 1, len(pred_labels)):
					y_p_i, y_p_j = pred_labels[i], pred_labels[j]
					y_t_i, y_t_j = true_labels[i], true_labels[j]
					if y_p_i == y_p_j and y_t_i == y_t_j:
						TP += 1
					elif y_p_i == y_p_j and y_t_i != y_t_j:
						FP += 1
					elif y_p_i != y_p_j and y_t_i != y_t_j:
						TN += 1
					elif y_p_i != y_p_j and y_t_i == y_t_j:
						FN += 1

			# TP / (TP + FP + FN)

			self._confusion_matrix[algo] = {
				'TP': TP,
				'FP': FP,
				'TN': TN,
				'FN': FN
			}
			print(f'Confusion matrix of algorithm: {algo}')
			print(self._confusion_matrix[algo])

	def plot_metrics_matching_sets(self, include_f1_score=True, include_precision=True, include_recall=True):

		if not include_f1_score and not include_precision and not include_recall:
			return

		df = []
		for algo in list(self._results):

			TP = self._confusion_matrix[algo]['TP']
			FP = self._confusion_matrix[algo]['FP']
			FN = self._confusion_matrix[algo]['FN']

			precision = TP / (TP + FP)
			recall = TP / (TP + FN)
			f1_score = 2 / (1 / precision + 1 / recall)

			if include_f1_score:
				df.append([algo, 'f1_score', f1_score])
			if include_precision:
				df.append([algo, 'precision', precision])
			if include_recall:
				df.append([algo, 'recall', recall])

		df = DataFrame(df, columns=['Algorithms', 'metrics', 'val'])
		df = df.pivot(index='metrics', columns='Algorithms', values='val')
		df.plot(kind='bar')
		plt.title('Metrics matching sets')
		plt.xticks(rotation=0)
		plt.legend(loc=(0.97, 0))
		plt.show()
		print(df)



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

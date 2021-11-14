import numpy as np
from utils.clustering_utils import assign_cluster, change_in_clusters, compute_labels


class KMeans:
    """
		implementation of K-means clustering algorithm

	"""
    centroids = None

    def __init__(self):
        pass

    def initialize_clusters(self, df, n_cluster, distance_metric, seed, naive=False):
        if naive:
            return df[np.random.choice(df.shape[0], n_cluster, replace=False), :]

        np.random.seed(seed)

        V = np.zeros((n_cluster, df.shape[1]))
        V[0, :] = df[np.random.choice(df.shape[0]), :]

        dists = np.zeros((df.shape[0], n_cluster - 1))
        for s in range(n_cluster - 1):
            for i, x in enumerate(df):
                dists[i, s] = distance_metric(x, V[s, :])
            dists_normalized = dists[:, :(s + 1)].min(axis=1)
            dists_normalized = dists_normalized / dists_normalized.sum()

            V[s + 1, :] = df[np.random.choice(df.shape[0], p=dists_normalized), :]

        return V

    def train(self, df, n_cluster, distance_metric, seed=4):
        """
            :param distance_metric -> 	should be a function which calculate distance
                                        might be l1(for k-medoids),l2(for k-means)
        """
        n_attributes = df.shape[1]

        V = self.initialize_clusters(df, n_cluster, distance_metric, seed)  # (n_cluster, n_attributes)
        V_old = 0.01 * V.copy()

        n_iter = 0

        aux_n_cluster = np.zeros(n_cluster)
        aux_clusters_coord = np.zeros((n_cluster, n_attributes))

        while change_in_clusters(V, V_old) and n_iter < 100:
            for i in range(df.shape[0]):
                x = df[i, :]
                cluster, distance = assign_cluster(x, V, distance_metric)
                aux_clusters_coord[cluster] += x
                aux_n_cluster[cluster] += 1

            V_old = V.copy()
            aux_n_cluster[aux_n_cluster == 0] = 1

            for k in range(aux_clusters_coord.shape[0]):
                V[k, :] = aux_clusters_coord[k, :] / aux_n_cluster[k]

            n_iter += 1

            aux_n_cluster = np.zeros(n_cluster)
            aux_clusters_coord = np.zeros((n_cluster, n_attributes))

        # print(f'Number of iterations: {n_iter}')
        self.centroids = V
        return V

    def predict(self, df, distance_metric):
        try:
            return compute_labels(df, self.centroids, distance_metric)
        except:
            raise Exception('No clusters trained')
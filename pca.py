import numpy as np



def _print_eigen_value_and_vectors(values, vectors):
	for val, vct in zip(values, vectors):
		print(f'[{val}] -> {vct}')


class PCA:
	def __init__(self):
		pass

	def fit_transform(self, df, desired_components):
		"""
		:param df: data frame with normalized fields
		:param desired_components: number of desired components
		:return: dataframe with reduce dimensionality
		"""
		# 1.3. 	Compute the d-dimensional mean vector (i.e., the means of every dimension of the whole data set)
		df_mean = df - np.mean(df, axis=0)

		# 1.4. 	Compute the covariance matrix of the whole data set. Show this information.
		df_covariance_matrix = np.cov(df_mean, rowvar=False)

		# 1.5. 	Calculate eigenvectors (e1, e2, ..., ed) and their corresponding eigenvalues of the covariance matrix.
		# 		Use numpy library. Write them in console.
		eigen_values, eigen_vectors = np.linalg.eigh(df_covariance_matrix)

		_print_eigen_value_and_vectors(values=eigen_values, vectors=eigen_vectors)

		# 1.6. 	Sort the eigenvectors by decreasing eigenvalues and
		# 		choose k eigenvectors with the largest eigenvalues to form a new d x k dimensional matrix (where every column represents an eigenvector).
		# 		Write the sorted eigenvectors and eigenvalues in console.
		sorted_index = np.argsort(eigen_values)[::-1]
		sorted_eigenvalue = eigen_values[sorted_index]
		sorted_eigenvectors = eigen_vectors[:, sorted_index]
		print("\n\n Sorted::\n")
		_print_eigen_value_and_vectors(values=sorted_eigenvalue, vectors=sorted_eigenvectors)

		# 1.7. 	Derive the new data set. Use this d x k eigenvector matrix to transform the samples onto the new subspace.
		eigenvector_subspace = sorted_eigenvectors[:, 0:desired_components]
		df_reduced = np.dot(eigenvector_subspace.transpose(), df_mean.transpose()).transpose()

		# 1.8. what we should do with reverted
		reverted = np.dot(eigenvector_subspace, df_reduced.transpose()).transpose() + df_mean

		return {
			'reduced': df_reduced,
			'covariance_matrix': df_covariance_matrix,
			'eigen_vectors': eigen_vectors,
			'eigen_values': eigen_values,
			'reverted': reverted
		}

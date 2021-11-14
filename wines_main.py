import pca
from pipelines.generic_pipeline import clean_numerical_data
from utils import file_utils
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#
# 1.1.	Read the .arff file and take the whole data set consisting of d-dimensional samples ignoring the class labels.
# 		Save the information in a matrix.
wines_raw_data = file_utils.load_arff('datasets/wine.arff')
wines_y = pd.DataFrame(wines_raw_data['class'].apply(lambda bts: int(bts)))
wines_clean_data = clean_numerical_data(wines_raw_data, ['a' + str(num) for num in range(1, 14)])


#
# 	todo choose 3 random features and plot things
#

#
# 1.2. Plot the original data set (choose two or three of its features to visualize it).
#
pca_result, rev_pca_result = pca.pca(wines_clean_data, 2)

pca_result2, _ = pca.pca(rev_pca_result, 2)

pca = PCA(n_components=2)
cc = pca.fit_transform(wines_clean_data)

k1, k11 = pca_result[wines_y['class'] == 1], cc[wines_y['class'] == 1]
k2, k22 = pca_result[wines_y['class'] == 2], cc[wines_y['class'] == 2]
k3, k33 = pca_result[wines_y['class'] == 3], cc[wines_y['class'] == 3]

plt.title('PCA')
plt.scatter(k1[:, 0], k1[:, 1], color='red')
plt.scatter(k2[:, 0], k2[:, 1], color='green')
plt.scatter(k3[:, 0], k3[:, 1], color='blue')
plt.show()

plt.title('sklearn PCA')
plt.scatter(k11[:, 0], k11[:, 1], color='red')
plt.scatter(k22[:, 0], k22[:, 1], color='green')
plt.scatter(k33[:, 0], k33[:, 1], color='blue')
plt.show()

k111 = pca_result2[wines_y['class'] == 1]
k222 = pca_result2[wines_y['class'] == 2]
k333 = pca_result2[wines_y['class'] == 3]

plt.title('Rev PCA')
plt.scatter(k111[:, 0], k111[:, 1], color='red')
plt.scatter(k222[:, 0], k222[:, 1], color='green')
plt.scatter(k333[:, 0], k333[:, 1], color='blue')
plt.show()


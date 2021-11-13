from utils import file_utils
from pipelines.generic_pipline import clean_numerical_data
import pandas as pd
import umap
import matplotlib.pyplot as plt

dataset_name = 'Wines'
dataset_filename = 'wine.arff'

# 1. load file
raw_data = file_utils.load_arff('datasets/wine.arff')
data_y = pd.DataFrame(raw_data['class'].apply(lambda bts: int(bts)))

# 2. clean data
clean_data = clean_numerical_data(raw_data, ['a' + str(num) for num in range(1, 14)])

fit = umap.UMAP()
u = fit.fit_transform(clean_data)

plt.scatter(u[:,0], u[:,1], c=data_y.to_numpy())
plt.title(f'UMAP embedding of {dataset_name} dataset');
plt.show()
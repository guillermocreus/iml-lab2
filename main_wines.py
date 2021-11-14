from utils import file_utils
from pipelines.generic_pipline import clean_numerical_data
from utils.aggregator import Aggregator
import pandas as pd
import umap
import matplotlib.pyplot as plt

agg = Aggregator()
dataset_name = 'Wines'
dataset_filename = 'wine.arff'

# 1. load file
raw_data = file_utils.load_arff('datasets/wine.arff')
data_y = pd.DataFrame(raw_data['class'].apply(lambda bts: int(bts)))

# 2. clean data
clean_data = clean_numerical_data(raw_data, ['a' + str(num) for num in range(1, 14)])

# 3. UMAP
u = agg.fit_UMAP(clean_data, data_y['class'], dataset_name=dataset_name)

_ = agg.fit_PCA(clean_data, 2, data_y['class'], dataset_name=dataset_name)

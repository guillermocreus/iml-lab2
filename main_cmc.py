from utils import file_utils
from pipelines.generic_pipline import clean_data_and_transform
from utils.aggregator import Aggregator
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt

agg = Aggregator()
dataset_name = 'CMC'
dataset_filename = 'cmc.arff'

class_column_name = 'class'
numeric_columns = ['wage', 'children']
columns_ordinal = ['weducation', 'heducation', 'wreligion', 'wworking', 'living_index', 'media_exposure']
columns_one_hot = ['hoccupation']

# 1. load file
raw_data = file_utils.load_arff(f'datasets/{dataset_filename}')
data_y = pd.DataFrame(raw_data[class_column_name].apply(lambda bts: int(bts)))

# 2. clean data
clean_data = clean_data_and_transform(raw_data, numeric_columns, columns_ordinal, columns_one_hot)

# 3. UMAP
u = agg.fit_UMAP(clean_data, data_y, dataset_name=dataset_name)

plt.scatter(clean_data[:, 0], clean_data[:, 1], c=data_y.to_numpy())
plt.title(f'2 dimensions of {dataset_name} dataset');
plt.show()

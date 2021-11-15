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

# umap parameters
umap_parameters = {
    'n_neighbors': 50,  # def = 15, higher --> global
    'min_dist': 0.9,  # def = 0.1, higher --> global
    'metric': 'euclidean'
}

# 1. load file
raw_data = file_utils.load_arff(f'datasets/{dataset_filename}')
data_y = pd.DataFrame(raw_data[class_column_name].apply(lambda bts: int(bts)))

# 2. clean data
clean_data = clean_data_and_transform(raw_data, numeric_columns, columns_ordinal, columns_one_hot)

# 3. evaluate methods
agg.evaluate(clean_data, 2, data_y[class_column_name], umap_parameters, dataset_name=dataset_name)

# 4. plot metrics
agg.plot_metrics_with_error()

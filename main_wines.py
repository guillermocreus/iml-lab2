from utils import file_utils
from pipelines.generic_pipline import clean_numerical_data
from utils.aggregator import Aggregator
import pandas as pd
import umap
import matplotlib.pyplot as plt

agg = Aggregator()
class_column_name = 'class'
dataset_name = 'Wines'
dataset_filename = 'wine.arff'

umap_parameters = {
    'n_neighbors': 20,  # def = 15, higher --> global
    'min_dist': 0.25,  # def = 0.1, higher --> global
    'metric': 'euclidean'
}

# 1. load file
raw_data = file_utils.load_arff(f'datasets/{dataset_filename}')
data_y = pd.DataFrame(raw_data['class'].apply(lambda bts: int(bts)))

# 2. clean data
clean_data = clean_numerical_data(raw_data, ['a' + str(num) for num in range(1, 14)])

# 3. evaluate methods
agg.evaluate(clean_data, 2, data_y[class_column_name], umap_parameters, dataset_name=dataset_name)

# 4. plot metrics
agg.plot_metrics_with_error()

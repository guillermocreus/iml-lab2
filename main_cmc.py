from pipelines.generic_pipeline import clean_numerical_data
from utils import file_utils
from pipelines.generic_pipline import clean_data_and_transform
from utils.aggregator import Aggregator
import pandas as pd
from utils.dataset_info import dataset_info
from utils.ploting_utils import plot_scatter

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
	'min_dist': 0.01,  # def = 0.1, higher --> global
	'metric': 'euclidean'
}

# 1. load file
raw_data = file_utils.load_arff(f'datasets/{dataset_filename}')
data_y = pd.DataFrame(raw_data[class_column_name].apply(lambda bts: int(bts)))


# 2. clean data
clean_data = clean_data_and_transform(raw_data, numeric_columns, columns_ordinal, columns_one_hot)
clean_data_ = pd.DataFrame(clean_data)
clean_data_.columns = numeric_columns + ['c' + str(num) for num in range(len(numeric_columns), clean_data.shape[1])]

# 2.1 information of the dataset
dataset_info(clean_data, data_y[class_column_name], dataset_name=dataset_name)

# 3. plot complete dataset
features_to_plot = ['wage', 'children']
plot_scatter(clean_data_[features_to_plot].to_numpy(), data_y[class_column_name],
             f'Scatter plot of dataset {dataset_name} based on {features_to_plot}')

# 4. evaluate methods
agg.evaluate(clean_data, 2, data_y[class_column_name], umap_parameters, dataset_name=dataset_name)

# 5. plot metrics
agg.plot_metrics_with_error()

# 6. external metrics
n_cluster_dict = {
	'complete': 3,
	'pca': 4,
	'umap': 3,
	'pca_sklearn': 4,
	'incremental_pca': 4
}

agg.compute_confusion_matrix(n_cluster_dict, data_y[class_column_name])
agg.plot_metrics_matching_sets()

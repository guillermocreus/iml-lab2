from pipelines.generic_pipeline import clean_numerical_data
from utils import file_utils
from pipelines.generic_pipline import clean_data_and_transform
from utils.aggregator import Aggregator
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
from utils.dataset_info import dataset_info
from utils.ploting_utils import plot_scatter

agg = Aggregator()
dataset_name = 'Hypothyroid'
dataset_filename = 'hypothyroid.arff'

class_column_name = 'Class'
numeric_columns = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
columns_ordinal = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
				   'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']
columns_one_hot = ['referral_source']

umap_parameters = {
	'n_neighbors': 50,  # def = 15, higher --> global
	'min_dist': 0.05,  # def = 0.1, higher --> global
	'metric': 'euclidean'
}

# 1. load file
raw_data = file_utils.load_arff(f'datasets/{dataset_filename}')
data_y = raw_data[class_column_name]

# 2. clean data
clean_data = clean_data_and_transform(raw_data, numeric_columns, columns_ordinal, columns_one_hot)
clean_data_ = pd.DataFrame(clean_data)
clean_data_.columns = numeric_columns + ['c' + str(num) for num in range(len(numeric_columns), clean_data.shape[1])]

data_y = pd.DataFrame(data_y.apply(lambda string: string.decode("utf-8", "ignore")))
tmp_data_y = OrdinalEncoder(categories='auto').fit_transform(data_y)
tmp_data_y = np.array([int(v[0]) for v in tmp_data_y])
data_y = pd.DataFrame({class_column_name: tmp_data_y})

# 2.1 information of the dataset
dataset_info(clean_data, data_y[class_column_name], dataset_name=dataset_name)

# 3. scatter clean data
features_to_plot = ['age', 'T3']
plot_scatter(clean_data_[features_to_plot].to_numpy(), data_y[class_column_name],
             f'Scatter plot of dataset {dataset_name} based on {features_to_plot}')

# 4. evaluate methods
agg.evaluate(clean_data, 2, data_y[class_column_name], umap_parameters, dataset_name=dataset_name)

# 5. plot metrics
agg.plot_metrics_with_error()

n_cluster_dict = {
	'complete': 3,
	'pca': 3,
	'umap': 3,
	'pca_sklearn': 3,
	'incremental_pca': 2
}

agg.compute_confusion_matrix(n_cluster_dict, data_y[class_column_name])
agg.plot_metrics_matching_sets()

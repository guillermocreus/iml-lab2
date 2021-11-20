from utils import file_utils
from pipelines.generic_pipline import clean_numerical_data
from utils.aggregator import Aggregator
import pandas as pd
from utils.dataset_info import dataset_info
from utils.ploting_utils import plot_scatter

agg = Aggregator()
class_column_name = 'class'
dataset_name = 'Wines'
dataset_filename = 'wine.arff'

umap_parameters = {
	'n_neighbors': 20,  # def = 15, higher --> global
	'min_dist': 0.05,  # def = 0.1, higher --> global
	'metric': 'euclidean'
}

# 1. load file
raw_data = file_utils.load_arff(f'datasets/{dataset_filename}')
data_y = pd.DataFrame(raw_data[class_column_name].apply(lambda bts: int(bts)))

# 2. clean data
clean_data = clean_numerical_data(raw_data, ['a' + str(num) for num in range(1, 14)])
clean_data_ = pd.DataFrame(clean_data)
clean_data_.columns = ['a' + str(num) for num in range(1, 14)]

# 2.1 information of the dataset
dataset_info(clean_data, data_y[class_column_name], dataset_name=dataset_name)

# 3. plot complete dataset
features_to_plot = ['a1', 'a2']
plot_scatter(clean_data_[features_to_plot].to_numpy(), data_y[class_column_name],
             f'Scatter plot of dataset{dataset_name} based on {features_to_plot}')

# 4. evaluate methods
agg.evaluate(clean_data, 2, data_y[class_column_name], umap_parameters, dataset_name=dataset_name)

# 5. plot metrics
agg.plot_metrics_with_error()

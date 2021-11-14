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

# 1. load file
raw_data = file_utils.load_arff('datasets/wine.arff')
data_y = pd.DataFrame(raw_data['class'].apply(lambda bts: int(bts)))

# 2. clean data
clean_data = clean_numerical_data(raw_data, ['a' + str(num) for num in range(1, 14)])

# 3. evaluate methods
agg.evaluate(clean_data, 2, data_y[class_column_name], dataset_name=dataset_name)

# 4. plot metrics
agg.plot_metrics_with_error()

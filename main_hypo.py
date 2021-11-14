from utils import file_utils
from pipelines.generic_pipline import clean_data_and_transform
from utils.aggregator import Aggregator
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt

agg = Aggregator()
dataset_name = 'Hypothyroid'
dataset_filename = 'hypothyroid.arff'

class_column_name = 'Class'
numeric_columns = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
columns_ordinal = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']
columns_one_hot = ['referral_source']


# 1. load file
raw_data = file_utils.load_arff(f'datasets/{dataset_filename}')
data_y = raw_data[class_column_name]

# 2. clean data
clean_data = clean_data_and_transform(raw_data, numeric_columns, columns_ordinal, columns_one_hot)
data_y = pd.DataFrame(data_y.apply(lambda string: string.decode("utf-8", "ignore")))

tmp_data_y = OrdinalEncoder(categories='auto').fit_transform(data_y)
tmp_data_y = np.array([int(v[0]) for v in tmp_data_y])
data_y = pd.DataFrame({class_column_name: tmp_data_y})

# 3. evaluate methods
agg.evaluate(clean_data, 2, data_y[class_column_name], dataset_name=dataset_name)

# 4. plot metrics
agg.plot_metrics_with_error()
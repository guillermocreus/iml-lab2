from sklearn.pipeline import Pipeline
from pipelines.transformers.byte_to_string_transformer import ByteToStringTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def clean_data_and_transform(data_frame, numeric_columns, string_columns_ordinal, string_columns_one_hot):
	x_copy = data_frame.copy()

	x_numeric_copy = x_copy[numeric_columns]
	num_pipeline = Pipeline([
		('missing_values_median_imputer', SimpleImputer(strategy="median")),
		('standard_scaler', StandardScaler()),
	])

	x_numeric_copy = num_pipeline.fit_transform(x_numeric_copy)

	x_string_ordinal_copy = x_copy[string_columns_ordinal]
	x_string_ordinal_copy = ByteToStringTransformer(string_columns_ordinal).transform(x_string_ordinal_copy)
	x_string_ordinal_copy = OrdinalEncoder(categories='auto').fit_transform(x_string_ordinal_copy)

	# if string_columns_one_hot:
	x_string_one_hot_copy = x_copy[string_columns_one_hot]
	x_string_one_hot_copy = ByteToStringTransformer(string_columns_one_hot).transform(x_string_one_hot_copy)
	# output is a SciPy sparse matrix, instead of a NumPy array
	# this values is matrix and Idk how to add it into matrix
	x_string_one_hot_copy = OneHotEncoder().fit_transform(x_string_one_hot_copy)

	aux = np.c_[x_numeric_copy, x_string_ordinal_copy, x_string_one_hot_copy.toarray()]
	# print(numeric_columns)
	# print(string_columns_ordinal)
	# print(string_columns_one_hot)
	# print(np.append(np.append(numeric_columns, string_columns_ordinal), string_columns_one_hot))
	# aux = pd.DataFrame(aux, columns=np.append(np.append(numeric_columns, string_columns_ordinal), string_columns_one_hot))
	# print(aux.head())
	return aux


def clean_numerical_data(data_frame, numeric_columns):
	x_copy = data_frame.copy()

	x_numeric_copy = x_copy[numeric_columns]
	num_pipeline = Pipeline([
		('missing_values_median_imputer', SimpleImputer(strategy="median")),
		('standard_scaler', StandardScaler()),
	])

	return num_pipeline.fit_transform(x_numeric_copy)

def clean_text_data(data_frame, string_columns_ordinal, string_columns_one_hot):
	x_copy = data_frame.copy()

	x_string_ordinal_copy = x_copy[string_columns_ordinal]
	x_string_ordinal_copy = ByteToStringTransformer(string_columns_ordinal).transform(x_string_ordinal_copy)
	x_string_ordinal_copy = OrdinalEncoder(categories='auto').fit_transform(x_string_ordinal_copy)

	x_string_one_hot_copy = x_copy[string_columns_one_hot]
	x_string_one_hot_copy = ByteToStringTransformer(string_columns_one_hot).transform(x_string_one_hot_copy)
	# output is a SciPy sparse matrix, instead of a NumPy array
	# this values is matrix and Idk how to add it into matrix
	x_string_one_hot_copy = OneHotEncoder().fit_transform(x_string_one_hot_copy)

	aux = np.c_[x_string_ordinal_copy, x_string_one_hot_copy.toarray()]

	return aux
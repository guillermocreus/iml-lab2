from scipy.io.arff import loadarff
import pandas as pd


def load_arff(path, as_data_frame=True):
	"""
		loads file as pandas dataframe
			@:param path -> represent path to file
			@:param as_data_frame -> cast of object
	"""
	raw_arff = loadarff(path)
	return pd.DataFrame(raw_arff[0]) if as_data_frame else raw_arff
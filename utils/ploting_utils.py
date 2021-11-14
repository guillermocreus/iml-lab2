import matplotlib.colors as colors
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

colors = list(colors._colors_full_map.values())


def plot_scatter(data: DataFrame, labels: Series, title: str):
	for current_label in labels.unique():
		current_category = data[labels == current_label]
		plt.scatter(current_category[:, 0], current_category[:, 1], color=colors[current_label + 8])
	plt.title(title)
	plt.show()


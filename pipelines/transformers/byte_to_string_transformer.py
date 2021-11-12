from sklearn.base import BaseEstimator, TransformerMixin


class ByteToStringTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, string_fields):
		self.string_fields = string_fields

	def fit(self, x, y=None):
		return self

	def transform(self, x, y=None, **fit_params):
		new_x = x.copy()
		for string_field in self.string_fields:
			new_x[string_field] = new_x[string_field].apply(lambda bts: bts.decode('utf8'))
		return new_x

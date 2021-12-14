from hypernetx import *
import pandas as pd
import numpy as np
from collections import OrderedDict

class Entity:

	def __init__(self, data=None, arr=None, labels=None,):
		self.properties = {}

		if data is not None:
			self._init_from_data(data)
		elif arr is not None:
			self._init_from_incidence(arr)
		else:
			self._init_empty(labels)

	def _init_from_data(self, data):

		if isinstance(data, pd.DataFrame):
			self._data, self._labels = _data_from_dataframe(data)
		elif isinstance(data, np.ndarray) and data.ndim == 2:
			self._data = data

	def _init_from_incidence(self,arr):
		pass

	def _init_empty(self,labels):
		pass

def _data_from_dataframe(df):
	data = np.empty(df.shape,dtype=int)
	labels = OrderedDict()
	# encode each column
	for i,col in enumerate(df.columns):
		# get unique values, and encode data column by unique value indices
    	unique_vals, encoding = np.unique(df[col].to_numpy(dtype=str), return_inverse=True)
    	data[:,i] = encoding

    	# add unique values to label dict
    	labels.update({col:unique_vals})
    return data, labels
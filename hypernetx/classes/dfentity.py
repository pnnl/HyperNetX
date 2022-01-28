from hypernetx import *
import pandas as pd
import numpy as np
from collections import OrderedDict

class StaticEntity(object):

	def __init__(
		self,
		data, # DataFrame, Dict of Lists, List of Lists, or np array
		weights=None, # array-like of values corresponding to rows of data
	):
		if isinstance(data, pd.DataFrame):
			# dataframe case
			self._data = data
		elif isinstance(data, dict):
			# dict of lists case
			k = sum([[i]*len(data[i]) for i in data],[])
			v = sum(data.values(),[])
			self._data = pd.DataFrame({0:k, 1:v})
		elif isinstance(data, list):
			# list of lists case
			k = sum([i]*len(data[i]) for i in range(len(data)), [])
			v = sum(data,[])
			self._data = pd.DataFrame({0:k, 1:v})
		elif isinstance(data, np.ndarray) and data.ndim == 2:
			self._data = pd.DataFrame(data)


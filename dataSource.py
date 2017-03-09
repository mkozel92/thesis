import pandas as pd
from numpy.random import randn
import numpy as np

class dataSource(object):
  def __init__(self,source):
    pass

  #this utterly useless thing might get useful with more data types and sources
  def get_data(self):
    return pd.read_csv('/home/michal/Downloads/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_3.csv',
                       index_col='timestamp')

  def generate_lin_data(self, x, x_var, count, slope):
    x_std = np.sqrt(x_var)
    xs = []
    orig = x
    for i in range(count):
      x = orig + i*slope + (randn() * x_std)
      xs.append(x)
    return np.array(xs)
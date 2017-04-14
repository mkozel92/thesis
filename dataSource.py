import pandas as pd
from numpy.random import random, randint, randn
import numpy as np

class dataSource(object):
  def __init__(self,source):
    pass

  #this utterly useless thing might get useful with more data types and sources
  # '/home/michal/Downloads/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_3.csv'
  def get_data(self,location):
    return pd.read_csv(location,index_col='timestamp')

  def generate_lin_data(self, x, x_var, count, slope, anomaly_probability):
    x_std = np.sqrt(x_var)
    xs = []
    orig = x
    for i in range(count):
      x = orig + i*slope + (randn() * x_std)
      if random() < anomaly_probability:
        x *= randint(-10,11)
      xs.append(x)
    return np.array(xs)

  def generate_sin_data(self, x_height, x_range, x_period, x_var, count, anomaly_probability):
    x_std = np.sqrt(x_var)
    xs = []
    for i in range(count):
      x = x_range * np.sin(i/x_period) + (randn() * x_std) + x_height
      if random() < anomaly_probability:
        x += x_range * randint(-1,2)
      xs.append(x)
    return np.array(xs)


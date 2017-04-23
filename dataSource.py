import pandas as pd
from numpy.random import random, randint, randn
import numpy as np

class dataSource(object):
  def __init__(self,source):
    pass

  @staticmethod
  def get_data(location):
    return pd.read_csv(location,index_col='timestamp')

  @staticmethod
  def generate_lin_data(x, x_var, count, slope, anomaly_probability):
    x_std = np.sqrt(x_var)
    xs = []
    orig = x
    for i in range(count):
      x = orig + i*slope + (randn() * x_std)
      if random() < anomaly_probability:
        x *= randint(-10,11)
      xs.append(x)
    return np.array(xs)

  @staticmethod
  def generate_sin_data(x_height, x_range, x_period, x_var, count, anomaly_probability):
    x_std = np.sqrt(x_var)
    xs = []
    for i in range(count):
      x = x_range * np.sin(i/x_period) + (randn() * x_std) + x_height
      if random() < anomaly_probability:
        x += x_range * randint(-1,2)
      xs.append(x)
    return np.array(xs)


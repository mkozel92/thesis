import pandas as pd

class dataSource(object):
  def __init__(self,source):
    pass

  #this utterly useless thing might get useful with more data types and sources
  def get_data(self):
    return pd.read_csv('/home/michal/Downloads/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv')
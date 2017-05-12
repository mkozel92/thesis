import pandas as pd
import dataSource as ds
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from Detectors.RNNDetector import RNNDetector


dsource = ds.dataSource('')


def visualize(results):
  res = pd.DataFrame(results)
  data = res.loc[:,['real','prediction']]
  a_scores = res.loc[:,['anomaly_score']]
  fig, axes = plt.subplots(nrows=2, ncols=1)
  data.plot(ax=axes[0])
  a_scores.plot(ax=axes[1])
  plt.show()

def nab_tst():
  d = dsource.get_data('../data/artificialWithAnomaly/art_daily_flatmiddle.csv')
  # d = dsource.get_data('data/artificialWithAnomaly/art_daily_jumpsdown.csv')
  # d = dsource.get_data('data/artificialWithAnomaly/art_increase_spike_density.csv')
  # d = dsource.get_data('data/realTweets/Twitter_volume_GOOG.csv')


  print (type(d))
  rd = RNNDetector(d[:1651],1)
  run(d['value'], rd)

def run(data, detector):
  results = []
  for x in data:
    results.append(detector.handle_record({'value': x}))
  visualize(results)

def main():
  nab_tst()


if __name__ == '__main__':
  main()
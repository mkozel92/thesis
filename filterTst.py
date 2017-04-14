import dataSource as ds
from KalmanDetector import KalmanDetector
import numpy as np
from utils import visualize


dsource = ds.dataSource('')

def sin_tst_1():
  d = dsource.generate_sin_data(30,30,100, 20, 2000, 0.002)

  kd = KalmanDetector([0.1, 0], [100], [1000], 0.001)
  run(d, kd)

def sin_tst_2():
  d = dsource.generate_sin_data(30,30,10, 0, 800, 0.01)

  kd = KalmanDetector([0.1, 0], [100], [1000], 1)
  run(d, kd)

def linear_tst_1():
  d = dsource.generate_lin_data(1, 3, 1000, 0, 0.01)
  d = np.append(d, dsource.generate_lin_data(1, 30, 1000, 0, 0.01))

  kd = KalmanDetector([0.1, 0], [100], [1000], 0.001)
  run(d, kd)

def linear_tst_2():
  d = dsource.generate_lin_data(1, 3, 200, 0.5,0.0)
  d = np.append(d,dsource.generate_lin_data(60,3,200,0.5,0.0))

  kd = KalmanDetector([0.1, 0], [100], [1000], 0.01)
  run(d,kd)

def run(data, k_detector):
  results = []
  for x in data:
    results.append(k_detector.handle_record({'value': x}))
  visualize(results)

def main():
  linear_tst_1()
  linear_tst_2()
  sin_tst_1()
  sin_tst_2()

if __name__ == '__main__':
  main()
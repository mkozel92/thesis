from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')

def create_filter(x, P, R, Q=0.1, dt=1.0):
  kf = KalmanFilter(dim_x=2, dim_z=1)
  kf.x = np.array([x[0], x[1]])
  kf.F = np.array([[1., dt],
                   [0., 1.]])  # state transition matrix
  kf.H = np.array([[1., 0]])  # Measurement function
  kf.R *= R
  if np.isscalar(P):
    kf.P *= P  # covariance matrix
  else:
    kf.P[:] = P
  if np.isscalar(Q):
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
  else:
    kf.Q[:] = Q
  return kf


def visualize(results):
  res = pd.DataFrame(results)
  data = res.loc[:,['real','prediction']]
  a_scores = res.loc[:,['anomaly_score']]
  fig, axes = plt.subplots(nrows=2, ncols=1)
  data.plot(ax=axes[0])
  a_scores.plot(ax=axes[1])
  plt.show()

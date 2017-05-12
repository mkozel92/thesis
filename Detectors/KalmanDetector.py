from OnlineMethods import OnlineVariance_2, ExponentialMovingAverage, Variance
from utils import create_filter
from filterpy.common import dot3
from numpy.linalg import inv
import numpy as np

class KalmanDetector(object):

  def __init__(self, x, P, R, Q=0.01, dt=1.0, eps_max = 1000, Q_scale_factor = 10):
    self.eps_max = eps_max
    self.Q_scale_factor = Q_scale_factor
    self.count = 0

    self.kf = create_filter(x, P, R, Q, dt)
    self.variance = OnlineVariance_2(25, [10])
    self.variance_2 = Variance(10,[])
    self.anomaly_ma = ExponentialMovingAverage(20,[0])

  def handle_record(self, inputData):
    self.kf.predict()
    prediction = self.kf.x[0]
    anomaly_score = abs(prediction - inputData["value"])/\
                    self.kf.P[0][0]
                    # (np.sqrt(self.variance.variance))
    self.anomaly_ma.add(anomaly_score)
    self.variance.add(inputData["value"], self.kf.x[0])
    self.variance_2.add(self.kf.x[0])
    self.kf.update(inputData["value"])

    #adapt filter
    y, S = self.kf.y, self.kf.S
    eps = dot3(y.T, inv(self.kf.S), y)

    if eps > self.eps_max:
        self.kf.Q *= self.Q_scale_factor
        self.count += 1
    elif self.count > 0:
        self.kf.Q /= self.Q_scale_factor
        self.count -= 1

    a_mean = self.anomaly_ma.mean + self.anomaly_ma.mean + np.sqrt(self.variance_2.variance)/150
    return {"anomaly_score" :anomaly_score,"anomaly_ma": a_mean,"is_a": (a_mean < anomaly_score), "prediction" :prediction, "real": inputData["value"]}
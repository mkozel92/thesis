from OnlineMethods import OnlineVariance_2, ExponentialMovingAverage
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
    self.variance = OnlineVariance_2(25, [1])
    self.change_point_score_avg = ExponentialMovingAverage(25,[10])

  def handle_record(self, inputData):
    self.kf.predict()
    prediction = self.kf.x[0]
    anomaly_score = abs(prediction - inputData["value"])/\
                    np.sqrt(self.variance.variance)
                    # np.sqrt(self.kf.P[0][0])
    self.variance.add(inputData["value"], self.kf.x[0])
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

    return {"anomaly_score" :anomaly_score, "prediction" :prediction, "real": inputData["value"]}
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.common import dot3
from filterpy.kalman import KalmanFilter
from numpy.linalg import inv
from copy import deepcopy

from ChangeDetector.strategy.base import ChangeDetectorStrategy
from ChangeDetector.strategy.utils import OnlineVariance_2, Variance, ExponentialMovingAverage


class KalmanDetector(ChangeDetectorStrategy):

    def __init__(self):
        pass

    def create_model(self, x = [0, 0], P = [10], R = [10], Q = 0.01, dt = 1, adaptive = False, eps_max = 1000,
                     q_scale_factor = 10):

        self.adaptive = adaptive
        self.eps_max = eps_max
        self.q_scale_factor = q_scale_factor

        self.init = False
        self.count = 0
        self.kf = self.create_filter(x, P, R, Q, dt)

        self.variance = OnlineVariance_2(10, [10])
        self.variance_2 = Variance(13, [])
        self.anomaly_ma = ExponentialMovingAverage(10, [0])


    def update(self, data, t=None):
        if not self.init:
            self.kf.x[0] = data[0]
            self.init = True

        for x in data:
            self.kf.predict()
            self.kf.update(x)

            if self.adaptive:
                y, S = self.kf.y, self.kf.S
                eps = dot3(y.T, inv(self.kf.S), y)
                if eps > self.eps_max:
                    self.kf.Q *= self.q_scale_factor
                    self.count += 1
                elif self.count > 0:
                    self.kf.Q /= self.q_scale_factor
                    self.count -= 1


    def detect(self, data, t=None):
        P_current, x_current, Q_current = deepcopy(self.kf.P), deepcopy(self.kf.x), self.kf.Q
        variance,anomaly_ma = deepcopy(self.variance_2), deepcopy(self.anomaly_ma)
        anomalies = []
        for i, x in enumerate(data):
            self.kf.predict()
            prediction = np.dot(self.kf.H, self.kf.x)
            anomaly_score = abs(prediction - x) / self.kf.P[0][0]
            self.anomaly_ma.add(anomaly_score)
            self.variance.add(x, prediction)
            self.variance_2.add(prediction)
            self.update([x])
            a_mean = self.anomaly_ma.mean + np.sqrt(self.variance_2.variance)
            if a_mean < anomaly_score:
                anomalies.append(i)
        self.kf.P, self.kf.x, self.kf.Q = P_current, x_current, Q_current
        self.variance_2, self.anomaly_ma = variance, anomaly_ma
        return anomalies

    def predict(self, data, t=None):
        P_current, x_current = deepcopy(self.kf.P), deepcopy(self.kf.x)
        predictions = []
        for i, x in enumerate(data):
            self.kf.predict()
            prediction = np.dot(self.kf.H, self.kf.x)[0]
            predictions.append(prediction)
            self.kf.update(x)
        self.kf.P, self.kf.x = P_current, x_current
        return predictions


    @staticmethod
    def create_filter(x, P, R, Q=0.1, dt=1.0):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([x[0], x[1]])
        kf.F = np.array([[1., dt], [0., 1.]])
        kf.H = np.array([[1., 0]])
        kf.R *= R
        if np.isscalar(P):
            kf.P *= P
        else:
            kf.P[:] = P
        if np.isscalar(Q):
            kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
        else:
            kf.Q[:] = Q
        return kf

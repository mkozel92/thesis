import dataSource as ds
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from OnlineMethods import ExponentialMovingAverage, OnlineVariance_2


def create_filter(x, P, R, Q=30, dt=1.0):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]])  # location and velocity
    kf.F = np.array([[1., dt],
                     [0., 1.]])  # state transition matrix
    kf.H = np.array([[1., 0]])  # Measurement function
    kf.R *= R  # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P  # covariance matrix
    else:
        kf.P[:] = P  # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf


filter = create_filter([0.1, 0], [100], [1000])
dsource = ds.dataSource('')
ex = ExponentialMovingAverage(100, [1])

variance = OnlineVariance_2(50,[1])
d = dsource.generate_lin_data(1,3,300,0)
d = np.append(d,dsource.generate_lin_data(5,3,300,-0))
d = np.append(d,dsource.generate_lin_data(5,3,300,-0.1))
d = np.append(d,dsource.generate_lin_data(-25,10,500,0))
d= dsource.get_data()['value']
# print (d
# plt.figure()
# d['is_anomaly'] *= d['value'].max()
# print (d)
n_iter = d.size
# print (n_iter)

predictions, xs, cov, filtered, ema, varis, pred_vars = [], [], [], [], [], [], []
for z in d:
    filter.predict()
    predictions.append(filter.x[0])
    ex.add(z)
    variance.add(z, filter.x[0])
    ema.append(ex.mean)
    varis.append(variance.variance)
    filter.update([z])
    filtered.append(filter.x[0])
    xs.append([z])
    cov.append(filter.x[1])
ema, filtered, varis, pred_vars = np.array(ema), np.array(filtered), np.array(varis), np.array(pred_vars)
xs, cov = np.array(xs).flatten(), np.array(cov)
# print
print ("varis")
print (varis)
# print (xs)
# exit(0)
plt.figure(1)
plt.subplot(411)
plt.plot(range(n_iter), xs, '.' ,linewidth=0.1)
plt.plot(range(n_iter), predictions)
plt.plot(range(n_iter), filtered)
plt.grid(True)
plt.subplot(412)
plt.grid(True)
plt.plot(range(n_iter), predictions-xs)
plt.subplot(413)
plt.grid(True)
plt.plot(range(n_iter), cov)
plt.subplot(414)
plt.grid(True)
plt.plot(range(n_iter), varis)
# plt.plot(range(n_iter), cov)
# plt.plot(range(n_iter), filtered)

plt.show()

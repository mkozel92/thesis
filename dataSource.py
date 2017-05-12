from numpy.random import random, randint, randn, choice
import numpy as np

class DataGenerator(object):

    def __init__(self):
        pass

    def generate_lin_data(self,x=10, x_var=10, count=10, slope=0, anomaly_probability=0.01):
        x_std = np.sqrt(x_var)
        xs,anomalies = [],[]
        orig = x
        for i in range(count):
            x = orig + i*slope + (randn() * x_std)
            if random() < anomaly_probability:
                anomalies.append(i)
                x += choice([randint(-8*x_std,-4*x_std),randint(4*x_std,8*x_std)])
            xs.append(x)
        return np.array(xs), np.array(anomalies)


    def generate_sin_data(self,x_height=10, x_range=10, x_period=10, x_var=10, count=10, anomaly_probability=0.01):
        x_std = np.sqrt(x_var)
        xs, anomalies = [],[]
        for i in range(count):
            x = x_range * np.sin(i/x_period) + (randn() * x_std) + x_height
            if random() < anomaly_probability:
                x += choice([randint(-8*x_std,-4*x_std),randint(4*x_std,8*x_std)])
                anomalies.append(i)
            xs.append(x)
        return np.array(xs), np.array(anomalies)


    def generate_change_point_data(self,*args):
        x_final, a_final = np.array([]), np.array([])
        for f in args:
            x, a = f
            a = np.append(a, x.size)
            a_final = np.append(a_final, a+x_final.size)
            x_final = np.append(x_final, x)

        return x_final, a_final


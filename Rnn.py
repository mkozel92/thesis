from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from copy import deepcopy

from ChangeDetector.strategy.base import ChangeDetectorStrategy
from ChangeDetector.strategy.utils import ExponentialMovingAverage, Variance


class RNNDetector(ChangeDetectorStrategy):

    def __init__(self):
        pass

    def create_model(self, look_back = 5):

        self.look_back = look_back

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(1, self.look_back)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.data = np.array([0] * self.look_back)
        self.variance_2 = Variance(10, [])
        self.anomaly_ma = ExponentialMovingAverage(20, [0])


    def update(self, data, t=None):
        d = pd.DataFrame(data)
        trainX, trainY = self.prepare_data(d, self.look_back)
        self.model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=0, shuffle=False)


    def detect(self, data, t=None):
        anomalies = []
        d = self.data
        variance, anomaly_ma = deepcopy(self.variance_2), deepcopy(self.anomaly_ma)

        for i,x in enumerate(data):
            predict = self.model.predict(np.reshape(self.data, (1, 1, self.data.shape[0])))
            predict = self.scaler.inverse_transform(predict)

            a = self.scaler.transform(np.reshape(x, (1, -1)))
            self.data = np.append(self.data[1:], a)

            prediction = predict[0, 0]
            anomaly_score = abs(prediction - x)

            self.anomaly_ma.add(anomaly_score)
            self.variance_2.add(x)

            a_mean = self.anomaly_ma.mean + self.anomaly_ma.mean + np.sqrt(self.variance_2.variance) / 10
            if a_mean < anomaly_score:
                anomalies.append(i)

        self.data = d
        self.variance_2, self.anomaly_ma = variance, anomaly_ma
        return anomalies


    def predict(self, data, t=None):
        predictions = []
        d = self.data
        for i,x in enumerate(data):
            predict = self.model.predict(np.reshape(self.data, (1, 1, self.data.shape[0])))
            predict = self.scaler.inverse_transform(predict)

            a = self.scaler.transform(np.reshape(x, (1, -1)))
            self.data = np.append(self.data[1:], a)

            prediction = predict[0, 0]
            predictions.append(prediction)

        self.data = d
        return predictions

    def prepare_data(self, dataset, look_back):
        d = dataset.astype('float32')
        d = self.scaler.fit_transform(d)

        dataX, dataY = [], []
        for i in range(len(d) - look_back - 1):
            a = d[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(d[i + look_back, 0])

        dataX, dataY = np.array(dataX), np.array(dataY)
        dataX = np.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))
        return dataX, dataY

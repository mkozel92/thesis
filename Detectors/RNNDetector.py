from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class RNNDetector(object):

  def __init__(self, train_data, look_back):
    self.scaler = MinMaxScaler(feature_range=(0, 1))
    trainX, trainY = self.prepare_data(train_data, look_back)
    self.model = self.build_model(trainX, trainY, look_back)
    self.data = np.array([0]*look_back)

  @staticmethod
  def build_model(trainX, trainY, look_back):
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back), stateful=True, batch_size=1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=1000, batch_size=1, verbose=2, shuffle=False)
    return model

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


  def handle_record(self, inputData):
    predict = self.model.predict(np.reshape(self.data, (1, 1, self.data.shape[0])))
    predict = self.scaler.inverse_transform(predict)

    a = self.scaler.transform(np.reshape([inputData['value']], (1, -1)))
    self.data = np.append(self.data[1:], a)

    prediction = predict[0,0]
    anomaly_score = abs(prediction - inputData['value'])
    return {"anomaly_score" :anomaly_score, "prediction" : prediction, "real": inputData["value"]}
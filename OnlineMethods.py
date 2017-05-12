import numpy as np

#weee some time-series
#I really shoud do this in pandas df
#compute mean of inital list and updates mean with every new added value
class OnlineMean(object):

  def __init__(self,initial = []):
    self.n = len(initial)
    self.mean = 0 if self.n == 0 else sum(initial)/self.n

  def add(self,x):
    self.n += 1
    self.mean += (x - self.mean)/self.n


#moving average over a fixed sized window with rememberin current elements
#offline in pandas for whole df: df['ma'] = pd.rolling_mean(df['value'],window = 42)
class MovingAverage(object):

  def __init__(self, window_size, initial = []):
    self.window_size = window_size
    self.elements = initial if len(initial) <= window_size else initial[-window_size:]
    self.mean = None if len(initial) < window_size else np.mean(initial[-window_size:])

  def add(self,x):
    if len(self.elements) >= self.window_size:
      del self.elements[0]
      self.elements.append(x)
      self.mean = np.mean(self.elements)
    else:
      self.elements.append(x)

class Variance(object):

      def __init__(self, window_size, initial=[]):
        self.window_size = window_size
        self.elements = initial if len(initial) <= window_size else initial[-window_size:]
        self.variance = 0 if len(initial) < window_size else np.var(initial[-window_size:])

      def add(self, x):
        if len(self.elements) >= self.window_size:
          del self.elements[0]
          self.elements.append(x)
          self.variance = np.var(self.elements)
        else:
          self.elements.append(x)
          self.variance = np.var(self.elements)

#Moving average without remembering elements -> so just an estimate
class ExponentialMovingAverage(object):

  def __init__(self, window_size, initial = []):
    self.window_size = window_size
    self.n = len(initial)
    self.mean = np.mean(initial) if self.n < window_size else np.mean(initial[-window_size:])

  def add(self,x):
    if self.n < self.window_size:
      self.n += 1
      self.mean += (x - self.mean) / self.n
    else:
      # mn+1 = m_n + a(x_n-m_n) = a*x_n + (1-a)*m_n = a*x_n + (1-a)*[a*x_n-1 + (1-a)*m_n-1] = a*x_n + (1-a)*a*x_n-1 + (1-a)^2*m_n-1....
      self.mean += (x - self.mean) / self.window_size

#compute variance of inital list and update with new elements
class OnlineVariance(object):

  def __init__(self, initial = []):
    self.variance = None if len(initial) == 0 else np.var(initial)
    self.om = OnlineMean(initial)
    self.m2 = 0 if not self.variance else self.om.n * self.variance

  def add(self,x):
    d = x - self.om.mean
    self.om.add(x)
    d2 = x - self.om.mean
    self.m2 += d * d2
    self.variance = self.m2/self.om.n

class OnlineVariance_2(object):
  def __init__(self, window_size, initial = []):
    self.window_size = window_size
    self.variance = None if len(initial) == 0 else np.var(initial[-self.window_size:])
    self.om = ExponentialMovingAverage(window_size, initial)
    self.m2 = 0 if not self.variance else self.om.n * self.variance

  def add(self,x,m):
    self.om.add(((x - m)**2))
    self.variance = self.om.mean

#compute covariance of inital lists and update with new elements
class OnlineCovariance(object):

  def __init__(self, initial_1 = [], initial_2 = []):
    if len(initial_1) != len(initial_2):
      raise Exception("list size discrepancy")
    self.covariance = np.cov(initial_2,initial_1)[0][1]
    self.m1 = np.mean(initial_1)
    self.m2 = np.mean(initial_2)
    self.n = len(initial_2)
    self.m12 = self.covariance * ((self.n -1)/ self.n)

  def add(self,x,y):
    self.n += 1
    d1 = (x - self.m1) / self.n
    self.m1 += d1
    d2 = (y - self.m2) / self.n
    self.m2 += d2
    self.m12 += (self.n - 1) * d1 * d2 - self.m12 / self.n
    self.covariance = self.n / (self.n -1) * self.m12
    print (self.m12)

#compute regression for initial lists and update regression parameters with new points
class OnlineRegression(object):

  def __init__(self, initial_1=[], initial_2=[]):
    if len(initial_1) != len(initial_2):
      raise Exception("list size discrepancy")
    self.cov = OnlineCovariance(initial_1, initial_2)
    self.xvar = OnlineVariance(initial_1)
    self.b = self.cov.covariance / self.xvar.variance
    self.a = self.cov.m2 - self.b * self.cov.m1

  def add(self,x,y):
    self.xvar.add(x)
    self.cov.add(x,y)
    self.b = self.cov.covariance / self.xvar.variance
    self.a = self.cov.m2 - self.b * self.cov.m1





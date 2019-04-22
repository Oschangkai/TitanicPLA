import copy
import numpy as np
import pandas as pd

def data():
  df = pd.read_csv('train.csv')

  feature_0 = copy.deepcopy(df['Pclass', 'Sex,' 'Age', 'SibSp', 'Parch'])
  feature_1 = copy.deepcopy(df['Pclass', 'Sex,' 'Age'])
  feature_2 = copy.deepcopy(df['Sex,' 'Age'])
  
  label = copy.deepcopy(df['Survived'])


class PLA(object):
  def __init__(self):
    self.W = np.random.uniform(-1, 1)

# Activation Function
  def sign(self, z):
    if z > 0:
      return 1
    elif z <= 0:
      return -1

  def err(self, prediction, truth):
    if prediction != truth:
      return True
    return False

if __name__ == '__name__':
  print('Hello')

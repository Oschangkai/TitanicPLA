import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_preprocessing(feature, na= True):
# 創建 Family = SibSp+Parch，分為 0, 1~4, 5~7, 8~10
  feature['Family'] = feature['SibSp'] + feature['Parch']
  feature['Family'] = feature['Family'].replace([1, 2, 3, 4], 1)
  feature['Family'] = feature['Family'].replace([5, 6, 7], 2)
  feature['Family'] = feature['Family'].replace([8, 9, 10], 3)
# SibSp 分為 0, 1~3, 4~6, 7~8
  feature['SibSp'] = feature['SibSp'].replace([1, 2, 3], 1)
  feature['SibSp'] = feature['SibSp'].replace([4, 5, 6], 2)
  feature['SibSp'] = feature['SibSp'].replace([7, 8], 3)
# Parch 分為 0, 1~3, 4~6
  feature['Parch'] = feature['Parch'].replace([1, 2, 3], 1)
  feature['Parch'] = feature['Parch'].replace([4, 5, 6], 2)
# Sex 把 male=0, female=1
  feature['Sex'] = feature['Sex'].map({'male': 0, 'female': 1}).astype(int)
# 填入 Age 中間值，並做 minMax 正規化
  if na == True:
    mid_age = feature['Age'].dropna().median()
    feature['Age'] = feature['Age'].fillna(value= mid_age)
  feature['Age'] = MinMaxScaler().fit_transform(feature[['Age']])


class PLA(object):

  def __init__(self, col):
#   Init weights and bias
    self.W = np.random.rand(1,col)
    self.b = np.random.randint(-10,10)

    self.pocket_err_rate = 100

# Activation Function
  def sign(self, z):

    if z > 0:
      return 1

    return -1

# 帶入
  def Forward(self, x):

    self.x = x

    self.z1 = np.dot(self.W, self.x)
    self.z1 += self.b

    self.out = self.sign(self.z1)

    return self.out

# 權重修正
  def Backward(self, y, rate= 1):
    self.y = y * rate
    self.W += np.dot(self.x, self.y)


  def err(self, prediction, truth):

    if prediction != truth:
      return True

    return False

  def pocket(self, err):
    if err < self.pocket_err_rate:
      self.pocketW = copy.deepcopy(self.W)
      self.pocket_err_rate = err


  def getWeight(self):
    print('pocket_err_rate: '+str(self.pocket_err_rate))

if __name__ == '__main__':
# 讀檔
  df = pd.read_csv('train.csv')
# 創建 feature
  feature_0 = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].copy(deep= True)
  feature_0_dropna = feature_0.copy(deep= True).dropna()
# 創建 label
  label = df['Survived'].copy(deep= True)
  label = label.replace([0], -1) # PLA 二元分法為 1, -1
# 資料清理
  data_preprocessing(feature_0)
  # data_preprocessing(feature_0_dropna, na= False)
# # 另外製作一些 feature 之後可以試試
#   feature_1 = feature_0[['Pclass', 'Sex', 'Age']].copy(deep= True)
#   feature_2 = feature_0[['Sex', 'Age']].copy(deep= True)
#   feature_1_dropna = feature_0_dropna[['Pclass', 'Sex', 'Age']].copy(deep= True)
#   feature_2_dropna = feature_0_dropna[['Sex', 'Age']].copy(deep= True)
# 迴圈上限
  max_iter = 1000
  learning_rate = 0.8
  pla = PLA(len(feature_0.columns))

  error_rate = []
  for iter in range(max_iter):

    err_count = 0

    for i in range(df.shape[0]):

      prediction = pla.Forward(feature_0.iloc[i])

      if pla.err(prediction, label.iloc[i]):
        err_count += 1
        pla.Backward(label.iloc[i], learning_rate)

    error_rate.append((100.0 * err_count) / df.shape[0])
    pla.pocket(error_rate[-1])

    print("iteration = %d, training error rate = %4.3f%%"
        % (iter, error_rate[-1]) )
    pla.getWeight()
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_cleaning(feature, na= True):
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
# 去除 Age N/A，或是填入中間值，並做 minMax 正規化
  if na == True:
    mid_age = feature['Age'].dropna().median()
    feature['Age'] = feature['Age'].fillna(value= mid_age)
  feature['Age'] = MinMaxScaler().fit_transform(feature[['Age']])


class PLA(object):
  def __init__(self):
    self.W = np.random.uniform(-1, 1)

# Activation Function
  def sign(self, z):
    if z > 0:
      return 1
    else:
      return -1

  def err(self, prediction, truth):
    if prediction != truth:
      return True
    return False

if __name__ == '__main__':
# 讀檔
  df = pd.read_csv('train.csv')
# 創建 feature
  feature_0 = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].copy(deep= True)
  feature_0_dropna = feature_0.copy(deep= True).dropna()
# 創建 label
  label = df['Survived'].copy(deep= True)
# 資料清理
  data_cleaning(feature_0)
  data_cleaning(feature_0_dropna, na= False)
# 另外製作一些 feature 之後可以試試
  feature_1 = feature_0[['Pclass', 'Sex', 'Age']].copy(deep= True)
  feature_2 = feature_0[['Sex', 'Age']].copy(deep= True)
  feature_1_dropna = feature_0_dropna[['Pclass', 'Sex', 'Age']].copy(deep= True)
  feature_2_dropna = feature_0_dropna[['Sex', 'Age']].copy(deep= True)
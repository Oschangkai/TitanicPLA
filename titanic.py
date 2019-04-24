import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_preprocessing(feature, na= True):
# Sex 把 male=0, female=1
  feature['Sex'] = feature['Sex'].map({'male': 0, 'female': 1}).astype(int)
# 填入 Age 中間值，並做 minMax 正規化
  if na == True:
    mid_age = feature['Age'].dropna().median()
    feature['Age'] = feature['Age'].fillna(value= mid_age)
  feature['Age'] = MinMaxScaler().fit_transform(feature[['Age']])

  # 若有兄弟姊妹欄位
  if 'SibSp' in feature.columns:
#   創建 Family = SibSp+Parch，分為 0, 1~4, 5~7, 8~10
    feature['Family'] = feature['SibSp'] + feature['Parch']
    feature['Family'] = feature['Family'].replace([1, 2, 3, 4], 1)
    feature['Family'] = feature['Family'].replace([5, 6, 7], 2)
    feature['Family'] = feature['Family'].replace([8, 9, 10], 3)
#   SibSp 分為 0, 1~3, 4~6, 7~8
    feature['SibSp'] = feature['SibSp'].replace([1, 2, 3], 1)
    feature['SibSp'] = feature['SibSp'].replace([4, 5, 6], 2)
    feature['SibSp'] = feature['SibSp'].replace([7, 8], 3)
#   Parch 分為 0, 1~3, 4~6
    feature['Parch'] = feature['Parch'].replace([1, 2, 3], 1)
    feature['Parch'] = feature['Parch'].replace([4, 5, 6], 2)




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
    self.b += y


  def err(self, prediction, truth):

    if prediction != truth:
      return True

    return False

# 儲存最佳 w
  def pocket(self, err):
    if err < self.pocket_err_rate:
      self.pocketW = copy.deepcopy(self.W)
      self.pocket_err_rate = err

# 取得目前 pocket 的 training 準確率
  def getPocketErrRate(self):
    print('pocket_err_rate: '+str(self.pocket_err_rate))
  
  def getPocketWeight(self):
    print('Final W: '+str(self.W))
    print('Final b:'+str(self.b))

if __name__ == '__main__':
# 讀檔
  df = pd.read_csv('train.csv')
# 創建 feature
  feature_0 = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].copy(deep= True)
  feature_0_dropna = feature_0.copy(deep= True).dropna() # 用來當其他的資料用
# 創建 label
  label = df['Survived'].copy(deep= True)
  label = label.replace([0], -1) # PLA 二元分法為 1, -1，1 為存活，-1 為死亡
# 資料清理
  data_preprocessing(feature_0)
  data_preprocessing(feature_0_dropna, na= False)
# 另外製作一些 feature 之後可以試試
  feature_1 = feature_0[['Pclass', 'Sex', 'Age']].copy(deep= True)
  feature_2 = feature_0[['Sex', 'Age']].copy(deep= True)
  feature_1_dropna = feature_0_dropna[['Pclass', 'Sex', 'Age']].copy(deep= True)
  feature_2_dropna = feature_0_dropna[['Sex', 'Age']].copy(deep= True)
# 迴圈上限
  max_iter = 500
  learning_rate = 0.8 # 測試用權重
  pla = PLA(len(feature_2_dropna.columns))

# Training
  error_rate = []
  for iter in range(max_iter):

    err_count = 0

    for i in range(feature_2_dropna.shape[0]):

      prediction = pla.Forward(feature_2_dropna.iloc[i])
#     如果有錯
      if pla.err(prediction, label.iloc[i]):
        err_count += 1
#       重新計算權重
        pla.Backward(label.iloc[i], learning_rate)
#   計算錯誤率
    error_rate.append((100.0 * err_count) / feature_2_dropna.shape[0])
    pla.pocket(error_rate[-1])

    print("iteration = %d, training error rate = %4.3f%%"
        % (iter, error_rate[-1]) )
    pla.getPocketErrRate()

  pla.getPocketWeight()

# Testing
  dft = pd.read_csv('test.csv')
  testing_0 = dft[['Sex', 'Age']].copy(deep= True)
  label_test = pd.read_csv('gender_submission.csv')['Survived']
  label_test = label_test.replace([0], -1)
  data_preprocessing(testing_0)

  err_count = 0
  predicts = []
  for i in range(dft.shape[0]):
    prediction = pla.Forward(testing_0.iloc[i])
    predicts.append(prediction)
    if pla.err(prediction, label_test.iloc[i]):
        err_count += 1
  PassengerId = pd.read_csv('gender_submission.csv')['PassengerId'].values
# kaggle 專用
  predicts_0 = [p if p > 0 else 0 for p in predicts] # 轉換 -1 為 0
  result = pd.DataFrame({"PassengerId" : PassengerId, "Survived" : predicts_0})
  result.to_csv('result.2.dropna.csv', index= 0)

  error_rate.append((100.0 * err_count) / dft.shape[0])
  print("testing error rate = %4.3f%%" % (error_rate[-1]))


  ## 把 Age = NA 的 drop 的結果，對於 training data 的 fitting 程度通常都低於 70%
  ## 只取 Age 和 Sex 對於(假) testing data 的 fitting 程度最高

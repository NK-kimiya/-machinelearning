import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

df = pd.read_csv('cinema.csv')
#欠損処理
numeric_columns = df.select_dtypes(include='number').columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
x = df.loc[:,'SNS1':'original']
t = df['sales']

#KFoldの処理で分割時の条件を指定
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=0)

#交差検証を行う
from sklearn.model_selection import cross_validate
model = LinearRegression()
result = cross_validate(model,x,t,cv=kf, scoring='r2',return_train_score=True)
print(result)

#平均値を計算
print(sum(result['test_score']/len(result['test_score'])))

#分類は正解データに偏りがないようにする
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=0)
print(skf)

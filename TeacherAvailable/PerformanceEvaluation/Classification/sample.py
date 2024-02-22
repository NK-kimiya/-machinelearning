#欠損値以外の前処理は省略
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

#データの準備
df = pd.read_csv('Survived.csv')
# 'Fare'列を数値型に変換し、変換できない場合はNaNにする
# 数値型の列に対してのみ欠損値を平均値で埋める
numeric_columns = df.select_dtypes(include='number').columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

x = df[['Pclass', 'Age']]
t = df['Survived']

#モデルの準備
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=2,random_state=0)
model.fit(x,t)

#再現率と適合率を一括で計算
from sklearn.metrics import classification_report
pred = model.predict(x)
out_put = classification_report(y_pred=pred,y_true=t,output_dict=True)
print(out_put)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
df = pd.read_csv('Boston.csv')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.mean())
df = df.drop([76],axis=0)

t =df[['PRICE']]#正解データ抜き出し
x =df.loc[:,['RM','PTRATIO','LSTAT']]#特徴量抜き出し

#標準化
sc = StandardScaler()
sc_x = sc.fit_transform(x)
sc2 = StandardScaler()
sc_t = sc2.fit_transform(t)

#累乗列と交互作用特徴量を一括追加する
pf = PolynomialFeatures(degree=2, include_bias=False)
pf_x = pf.fit_transform(sc_x)
print(pf_x.shape)

#列名を確認する処理
print(pf.get_feature_names_out())

#線形回帰で過学習が起こることを確認
x_train,x_test, y_train,y_test = train_test_split(pf_x,sc_t, test_size=0.3,random_state=0)
model = LinearRegression()
model.fit(x_train,y_train)
print(model.score(x_train,y_train))#訓練データの決定係数
print(model.score(x_test,y_test))#テストデータの決定係数

#ラッソ回帰モデルで過学習が起きてないか確認
x_train, x_test, y_train, y_test = train_test_split(pf_x,sc_t,test_size=0.3,random_state=0)

model = Lasso(alpha=0.1)
model.fit(x_train,y_train)

print(model.score(x_train,y_train))#訓練データの決定係数
print(model.score(x_test,y_test))#訓練データの決定係数

#回帰式の係数を確認
weight = model.coef_#係数を抜き出す
print(pd.Series(weight, index=pf.get_feature_names_out()))



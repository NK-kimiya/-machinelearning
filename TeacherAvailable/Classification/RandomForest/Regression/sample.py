import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#データの読み込み
df = pd.read_csv('cinema.csv')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.mean())
x = df.loc[:,'SNS1':'original']
t = df['sales']
x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)

#ランダムフォレスト回帰
from sklearn.ensemble import RandomForestRegressor
#100個のモデルで並列学習
model = RandomForestRegressor(random_state=0,n_estimators=100)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

#アダブースト回帰
# XGBoost回帰
from xgboost import XGBRegressor
model_xgb = XGBRegressor(random_state=0, n_estimators=100)
model_xgb.fit(x_train, y_train)
print("XGBoost Regression Score:", model_xgb.score(x_test, y_test))


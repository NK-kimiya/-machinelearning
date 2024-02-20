import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
df = pd.read_csv('Boston.csv')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.mean())
x = df.loc[:, 'ZN':'LSTAT']
t = df['PRICE']

x_train, x_test, y_train, y_test = train_test_split(x,t,test_size=0.3,random_state=0)

model = DecisionTreeRegressor(max_depth=10,random_state=0)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

#特徴量の重要度を参照する
print(pd.Series(model.feature_importances_, index=x.columns))




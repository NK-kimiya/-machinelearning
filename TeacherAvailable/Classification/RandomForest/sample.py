import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Survived.csv')
print(df.head(2))

#欠損値の穴埋め
jo1 = df['Pclass'] == 1
jo2 = df['Survived'] == 0
jo3 = df['Age'].isnull()
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 43

jo2 = df['Survived'] == 1
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 35

jo1 = df['Pclass'] == 2
jo2 = df['Survived'] == 0
jo3 = df['Age'].isnull()
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 26

jo2 = df['Survived'] == 1
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 20

jo1 = df['Pclass'] == 3
jo2 = df['Survived'] == 0
jo3 = df['Age'].isnull()
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 43

jo2 = df['Survived'] == 1
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 35

#決定木は外れ値と標準化は影響しない

#文字データの列を数値に変換
col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

x = df[col]
t = df['Survived']

dummy = pd.get_dummies(df['Gender'],drop_first=True)
x = pd.concat([x,dummy], axis=1)
print(x.head(2))

#ランダムフォレストのインポート
from sklearn.ensemble import RandomForestClassifier
x_train, x_test, y_train, y_test = train_test_split(x,t,test_size=0.2,random_state=0)
model = RandomForestClassifier(n_estimators=200,random_state=0)

#モデルの学習
model.fit(x_train, y_train)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))

#単純な決定木分類と比較
from sklearn import tree
model2 = tree.DecisionTreeClassifier(random_state=0)
model2.fit(x_train,y_train)

print(model2.score(x_train,y_train))
print(model2.score(x_test,y_test))

#特徴量の重要度を確認
importance = model.feature_importances_
print(pd.Series(importance, index=x_train.columns))




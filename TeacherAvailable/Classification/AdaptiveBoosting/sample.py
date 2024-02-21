from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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
base_model = DecisionTreeClassifier(random_state=0, max_depth=5)

#決定木を500個作成
# XGBoostのモデルを作成
model = XGBClassifier(n_estimators=500, max_depth=5, random_state=0, use_label_encoder=False, eval_metric='logloss')
model.fit(x_train, y_train)

# 訓練データとテストデータに対するモデルのスコアを表示
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))


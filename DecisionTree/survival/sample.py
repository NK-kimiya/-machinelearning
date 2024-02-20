import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle

#CSVファイルの読み込み
df = pd.read_csv('Survived.csv')
df2 = pd.read_csv('Survived.csv')#再学習用
print(df.head(2))

#正解データの集計ー不均衡データ
print(df['Survived'].value_counts())
'''
0 549
1 342
→0の値が549個、1の値が342個で不均衡よって、Survived列のデータは不均衡
'''

#欠損値を確認する
print(df.isnull().sum())

#データの行数と列数を確認
print(df.shape)

#Age列を平均値で穴埋め
age_mean = df['Age'].mean()# Age列の平均値を計算
df['Age'].fillna(age_mean, inplace=True) # Age列の欠損値を平均値で穴埋め


#Embarked列を最頻値で穴埋め
embraked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embraked_mode, inplace=True) # Embarked列の欠損値を最頻値で穴埋め

#特徴量xと正解データtに分割する
col = ['Pclass','Age','SibSp','Parch','Fare']
x = df[col]
t = df['Survived']

#訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x,t,test_size=0.2,random_state=0)

#モデルの作成と学習
model = tree.DecisionTreeClassifier(max_depth=5,random_state=0,class_weight='balanced')
model.fit(x_train, y_train)#学習

#正解率の計算
print(model.score(X=x_test, y= y_test))

#小グループ作成の基準となる列を指定
print(df.groupby('Survived')['Age'].mean())
print(df.groupby('Pclass')['Age'].mean())

#ピボットテーブル機能を使う
print(pd.pivot_table(df,index='Survived',columns = 'Pclass',values='Age'))

#Age列の欠損値の行を抜き出す
is_null = df2['Age'].isnull()

#Pclass1に関する埋め込み
df2.loc[(df2['Pclass'] == 1) & (df2['Survived'] == 0)
       &(is_null), 'Age'] = 43
df2.loc[(df2['Pclass'] == 1) & (df2['Survived'] == 1)
       &(is_null), 'Age'] = 35

#Pclass2に関する埋め込み
df2.loc[(df2['Pclass'] == 2) & (df2['Survived'] == 0)
       &(is_null), 'Age'] = 33
df2.loc[(df2['Pclass'] == 2) & (df2['Survived'] == 1)
       &(is_null), 'Age'] = 25

#Pclass3に関する埋め込み
df2.loc[(df2['Pclass'] == 3) & (df2['Survived'] == 0)
       &(is_null), 'Age'] = 26
df2.loc[(df2['Pclass'] == 3) & (df2['Survived'] == 1)
       &(is_null), 'Age'] = 20

#モデルに再学習
col2 = ['Pclass','Age','SibSp','Parch','Fare']
x2 = df2[col2]
t2 = df2['Survived']

#訓練データとテストデータに分割
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2,t2,test_size=0.2,random_state=0)
model = tree.DecisionTreeClassifier(max_depth=6,random_state=0,class_weight='balanced')#モデルの作成と学習
model.fit(x_train2, y_train2)#学習
print(model.score(X=x_test2, y= y_test2))

#平均値を求める
gender = df2.groupby('Gender')['Survived'].mean()
print(gender)

#文字列を数値に変換
male = pd.get_dummies(df2['Gender'])
print(male)

#2つのデータフレームを横方向に連結
x_tmp = pd.concat([x2,male],axis=1)
print(x_tmp)

#再々学習
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_tmp,t2,test_size=0.2,random_state=0)
model = tree.DecisionTreeClassifier(max_depth=5,random_state=0,class_weight='balanced')#モデルの作成と学習
model.fit(x_train2, y_train2)#学習
print(model.score(X=x_test2, y= y_test2))

#モデルの保存
with open('survived.pkl','wb') as f:
    pickle.dump(model,f)
    
#特徴量重要度を確認
print(model.feature_importances_)

#データフレームに変換
print(pd.DataFrame(model.feature_importances_,index = x_tmp.columns))


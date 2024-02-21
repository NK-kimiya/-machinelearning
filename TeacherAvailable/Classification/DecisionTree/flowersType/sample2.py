import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle
df = pd.read_csv('iris.csv')
#各列の平均値を計算
numeric_df = df.select_dtypes(include=['number'])# 数値型の列のみを選択
mean_values = numeric_df.mean()#選択した列の平均値を計算
print("列の平均値" + str(mean_values))
#平均値で欠損値を穴埋め
df2 = df.fillna(mean_values)
#欠損値があるか確認
print(df2.isnull().any(axis=0))
#特徴量と正解データを変数に代入
xcol = ['がく片長さ','がく片幅','花弁長さ','花弁幅']
x = df2[xcol]
t = df2['種類']
#モデルの作成
model = tree.DecisionTreeClassifier(max_depth = 2,random_state=0)
#モデルの学習
model.fit(x,t)
print(model.score(x,t))
#訓練データとテストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.3,random_state=0)
#訓練データで再学習
model.fit(x_train,y_train)
print("正解率：" + str(model.score(x_test,y_test)))#テストデータの予測結果と実際の答えが合致する正解率を計算
#モデルを保存
#with open('irismodel.pkl','wb') as f:
    #pickle.dump(model,f)
#分岐条件の列を決める
print(model.tree_.feature)
#分岐条件のしきい値を含む配列を返す
print(model.tree_.threshold)
#リーフに到達したデータの数を返す
print(model.tree_.value[1])#ノード番号1に到達したとき
print(model.tree_.value[3])#ノード番号3に到達したとき
print(model.tree_.value[4])#ノード番号4に到達したとき
#アヤメの種類とグループ番号の対応
print(model.classes_)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle
#ファイルの読み込み
df = pd.read_csv('cinema.csv')
print(df.head(3))

#欠損値の確認
print(df.isnull().any(axis=0))

#欠損値を平均で補完
numeric_df = df.select_dtypes(include=['number'])# 数値型の列のみを選択
df2 = df.fillna(numeric_df.mean())#選択した列の平均値を計算
print(df2.isnull().any(axis=0))#穴埋めができたか確認

#各特徴データと正解データの相関関係の散布図を表示
'''df2.plot(kind='scatter',x='SNS1',y='sales')
df2.plot(kind='scatter',x='SNS2',y='sales')
df2.plot(kind='scatter',x='actor',y='sales')
df2.plot(kind='scatter',x='original',y='sales')
plt.show()
'''
#外れ値を削除
no = df2[(df2['SNS2'] > 1000) & (df2['sales'] < 8500)].index#外れ値のデータを検索して、インデックスを取得
df3 = df2.drop(no,axis=0)

#データフレームを作成する
test = pd.DataFrame(
    {'Acolumn':[1,2,3],
     'Bcolumn':[4,5,6] 
    }
)

#Acolumn列の値が2未満だけの行を参照する
print(test[test['Acolumn'] < 2])

#特定した行からインデックスのみを取り出す
no = df[(df['SNS2'] > 1000) & (df['sales'] < 8500)].index#2つの条件で外れ値の行を特定する
print(no)

#インデックスが0の行を削除する
print(test.drop(0,axis=0))

#列を削除する
print(test.drop('Bcolumn',axis=1))

#外れ値を削除
df3 = df2.drop(no,axis=0)
print(df3.shape)

#df3から特徴量の変数xと正解データの変数tに分割
col = ['SNS1','SNS2','actor','original']
x = df3[col]#特徴量の取り出し
t = df3['sales']#正解データの取り出し

#インデックス2の行からSNS1列の値を取り出す
print(df3.loc[2,'SNS1'])

#特定のデータのみを参照する
index = [2,4,6]
col = ['SNS1','actor']#列名
print(df3.loc[index,col])

#スライス構文で連続した要素を参照
sample = [10,20,30,40]
print(sample[1:3])

#データフレームで複数のインデックスや列名を参照
print(df3.loc[0:3,:'actor'])#0行目以上、3行目以下、actor列より左(actor列含む)

#スライス構文で特徴量と正解データを取り出す
x=df3.loc[:,'SNS1':'original']#特徴量の取り出し
t = df3['sales']#正解ラベルの取り出し

#訓練データとテストデータに分割する
x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)

#回帰分析のモデルを作成
model = LinearRegression()

#モデルに学習
model.fit(x_train, y_train)

#興行収入を予測
new = [[150,700,300,0]]
print(model.predict(new))

#モデルのMAE(平均絶対誤差)を求める
pred = model.predict(x_test)#x_testデータを一括で予測
print(mean_absolute_error(y_pred=pred,y_true=y_test))

#決定係数を求める
model.score(x_test,y_test)

#モデルの保存
with open('cinema.pkl','wb') as f:
    pickle.dump(model,f)

#係数と切片の表示
print(model.coef_)#計算式の係数の表示
print(model.intercept_)#計算式の切片の表示

#列と係数を表示
tmp = pd.DataFrame(model.coef_)#データフレームの作成
tmp.index = x_train.columns
print(tmp)












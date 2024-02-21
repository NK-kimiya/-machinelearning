import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np  # NumPyをインポート
from sklearn.preprocessing import StandardScaler
import pickle

# CSVファイルからデータを読み込み、データフレームに格納
df = pd.read_csv('Boston.csv')
print(df.head(2))

#CRIME列にデータが何種類あるかを確認
print(df['CRIME'].value_counts())

#ダミー変数化した列を連結しCRIME列を削除
crime = pd.get_dummies(df['CRIME'], drop_first=True)
df = pd.concat([df,crime], axis=1)
df = df.drop(['CRIME'], axis=1)
print(df.head(2))

#訓練データ&検証データとテストデータに分割する
train_val, test= train_test_split(df,test_size = 0.2,random_state=0)

#train_valの欠損値を確認する
print(train_val.isnull().sum())

#欠損値を平均値で穴埋め
train_val_mean = train_val.mean()#各列の平均値の計算
train_val2 = train_val.fillna(train_val_mean)#平均値で穴埋め

#各特徴量の列とPRICE列の相関関係を示す散布図を描く
colame = train_val2.columns
colnames = train_val2.columns
numeric_features = train_val2.select_dtypes(include=[np.number]).columns.drop('PRICE')

n = len(numeric_features)
ncols = 3
nrows = n // ncols + (n % ncols > 0) 

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
fig.tight_layout(pad=4.0)

for i, name in enumerate(numeric_features):
    ax = axes.flatten()[i]
    ax.scatter(train_val2[name], train_val2['PRICE'])
    ax.set_title(f'{name} vs PRICE')
    ax.set_xlabel(name)
    ax.set_ylabel('PRICE')

if n % ncols != 0:
    for idx in range(i+1, nrows*ncols):
        axes.flatten()[idx].set_visible(False)

plt.show()

#RMの外れ値
out_line1 = train_val2[(train_val2['RM'] < 6) & (train_val2['PRICE'] > 40)].index

#PTRATIOの外れ値
out_line2 = train_val2[(train_val2['PTRATIO'] > 18) & (train_val2['PRICE'] > 40)].index

# インデックスとその長さを表示
print("out_line1:", out_line1, "Length:", len(out_line1))
print("out_line2:", out_line2, "Length:", len(out_line2))

#外れ値を削除する
train_val3 = train_val2.drop([76],axis=0)

#絞り込んだ列以外を取り除く
col = ['INDUS','NOX','RM','PTRATIO','LSTAT','PRICE']
train_val4 = train_val3[col]
print(train_val4.head(3))

#列同士の相関係数を調べる
print(train_val4.corr())

#各列とPRICE列との相関係数を見る
train_cor = train_val4.corr()['PRICE']
print(train_cor)

#abs関数で絶対値に変換
print(abs(1))
print(abs(-2))

#mapメソッドで要素に関数を適用する
se = pd.Series([1,-2,3,-4])
print(se.map(abs))

#相関行列のPRICE列との相関係数を絶対値に変換する
abs_cor = train_cor.map(abs)
print(abs_cor)

#要素を降順に並べる
print(abs_cor.sort_values(ascending=False))

#訓練データと検証データに分割する
col = ['RM','LSTAT','PTRATIO']
x = train_val4[col]
t = train_val4[['PRICE']]

x_train, x_val, y_train, y_val = train_test_split(x,t,test_size = 0.2,random_state=0)

#データの標準化
sc_model_x = StandardScaler()
sc_model_x.fit(x_train)#各列の平均値や標準偏差を調べる
sc_x = sc_model_x.transform(x_train)#各列のデータを標準化してsc_Xに代入
print(sc_x)

#平均値0を確認する
tmp_df = pd.DataFrame(sc_x, columns = x_train.columns)#データフレームにする
print(tmp_df.mean())

#標準偏差を計算する
print(tmp_df.std())

#正解データも標準化する
sc_model_y = StandardScaler()
sc_model_y.fit(y_train)
sc_y = sc_model_y.transform(y_train)

#標準化したデータで学習させる
model = LinearRegression()
model.fit(sc_x, sc_y)

#決定係数を求める
print(model.score(x_val, y_val))

#検証データも標準化する
sc_x_val = sc_model_x.transform(x_val)
sc_y_val = sc_model_y.transform(y_val)

print(model.score(sc_x_val, sc_y_val))

#learn関数の定義(訓練データと検証データの分割・訓練データと検証データの標準化)
def learn(x,t):
    x_train, x_val, y_train, y_val = train_test_split(x,t,test_size=0.2,random_state=0)
    #訓練データを標準化
    sc_model_x = StandardScaler()
    sc_model_y = StandardScaler()
    sc_model_x.fit(x_train)
    sc_x_train = sc_model_x.transform(x_train)
    sc_model_y.fit(y_train)
    sc_y_train = sc_model_y.transform(y_train)
    #学習
    model = LinearRegression()
    model.fit(sc_x_train,sc_y_train)
    #検証データを標準化
    sc_x_val = sc_model_x.transform(x_val)
    sc_y_val = sc_model_y.transform(y_val)
    #訓練データと検証データの決定係数計算
    train_score = model.score(sc_x_train, sc_y_train)
    val_score = model.score(sc_x_val, sc_y_val)
    
    return train_score,val_score

#learn関数を実行する
x = train_val3.loc[:,['RM','LSTAT','PTRATIO']]
t = train_val3[['PRICE']]
s1,s2 = learn(x,t)
print(s1,s2)

#特徴量にINDUS列を追加する
x = train_val3.loc[:,['RM','LSTAT','PTRATIO','INDUS']]
t = train_val3[['PRICE']]
s1,s2 = learn(x,t)
print(s1,s2)

#データフレームのRM列のデータを2乗する
print(x['RM'] ** 2)

#新しい列を特徴量に追加する
x['RM2'] = x['RM'] ** 2#RM2乗のシリーズを新しい列として追加
x = x.drop('INDUS', axis = 1)
print(x.head(2))

#再学習を行う
s1,s2 = learn(x,t)
print(s1,s2)

#LSTAT列、PTRATIO列で、新しい列を特徴量に追加
x['LSTAT2'] = x['LSTAT'] ** 2#LSTAT列の2乗を追加
s1,s2 = learn(x,t)
print(s1,s2)
x['PTRATIO2'] = x['PTRATIO'] ** 2
s1,s2 = learn(x,t)
print(s1,s2)

#相互作用特徴量を追加する
x['RM * LSTAT'] = x['RM'] * x['LSTAT']
print(x.head(2))

#特徴量を追加したので再学習を行う
s1,s2 = learn(x,t)
print(s1,s2)

#データの標準化後に再学習を行う
sc_model_x2 = StandardScaler()
sc_model_x2.fit(x)
sc_x = sc_model_x2.transform(x)

sc_model_y2 =StandardScaler()
sc_model_y2.fit(t)
sc_y = sc_model_y2.transform(t)
model = LinearRegression()
model.fit(sc_x,sc_y)

#テストデータの前処理
test2 = test.fillna(train_val.mean())#欠損値を平均値で補完
x_test = test2.loc[:,['RM','LSTAT','PTRATIO']]
y_test = test2[['PRICE']]

x_test['RM2'] = x_test['RM'] ** 2
x_test['LSTAT2'] = x_test['LSTAT'] ** 2
x_test['PTRATIO2'] = x_test['PTRATIO'] ** 2

x_test['RM * LSTAT'] = x_test['RM'] * x_test['LSTAT']
sc_x_test = sc_model_x2.transform(x_test)
sc_y_test = sc_model_y2.transform(y_test)
print(model.score(sc_x_test, sc_y_test))

#モデルを保存する
with open('boston.pkl','wb') as f:
    pickle.dump(model,f)
    
with open('boston_scx.pkl','wb') as f:
    pickle.dump(sc_model_x2,f)

with open('boston_scy.pkl','wb') as f:
    pickle.dump(sc_model_y2,f)

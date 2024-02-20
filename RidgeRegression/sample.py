import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
df = pd.read_csv('Boston.csv')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')
try:
    df = df.fillna(df.mean())
except Exception as e:
    print(f"エラーが発生しました: {e}")
df = df.drop([76],axis=0)

t =df[['PRICE']]#正解データ抜き出し
try:
    x =df.loc[:,['RM','PTRATIO','LSTAT']]#特徴量抜き出し
except Exception as e:
    # エラーが発生した場合の処理
    print(f"エラーが発生しました: {e}")


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

#リッジ回帰で過学習が起こるか確認
ridgeModel = Ridge(alpha=17.62)#モデルの定義
ridgeModel.fit(x_train,y_train)#学習
print(ridgeModel.score(x_train,y_train))
print(ridgeModel.score(x_test,y_test))

#正規化項の定数を0.01～20まで0.01刻みで検証するコード
maxScore = 0
MaxIndex = 0
#ridgeModel = Ridge(alpha=17.62)最適なalphaの値を調べる
for i in range(1,2001):
    num = i / 100
    ridgeModel = Ridge(random_state=0,alpha=num)
    ridgeModel.fit(x_train,y_train)
    result = ridgeModel.score(x_test,y_test)
    if result > maxScore:
        maxScore = result
        MaxIndex = num

print(MaxIndex,maxScore)

#重回帰とリッジ回帰の係数の大きさを比較する
print(sum(abs(model.coef_)[0]))#線形回帰の係数
print(sum(abs(ridgeModel.coef_)[0]))#リッジ回帰の合計

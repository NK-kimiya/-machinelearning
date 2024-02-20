import pandas as pd
import matplotlib.pyplot as plt
#bike.tsvを読み込む
df = pd.read_csv('bike.tsv',sep = '\t')
print(df.head(3))

#weather.csvを読み込む
weather = pd.read_csv('weather.csv', encoding='shift-jis')
print(weather)

#jsonファイルの読み込み
read_json = pd.read_json('temp.json')
temp = read_json.T#列と行の反転

#内部結合を行う
df2 = df.merge(weather,how='inner',on='weather_id')#dfのデータフレームのweather_idにweatherデータフレームのweather_idを結合
print(df2.head(2))

#weatherごとのcntの平均値を集計する
print(df2['cnt'].dtype)# 'cnt'列のデータ型を確認
df2['cnt'] = pd.to_numeric(df2['cnt'], errors='coerce')
df2['cnt'] = df2['cnt'].fillna(0)
print(f"'weather' dtype: {df2['weather'].dtype}")
result = df2[['weather', 'cnt']].groupby('weather').mean()
print(result)

#tempデータフレームの200行目付近を確認
print(temp.loc[199:201])

#2011-07-20を表示する
print(df2[df2['dteday'] == '2011-07-20'])

#外部結合を行う
df3 = df2.merge(temp, how='left', on='dteday')
print(df3[df3['dteday'] == '2011-07-20'])

#気温に関する折れ線グラフを作成する
plt.figure(figsize=(10, 6))  # グラフのサイズを設定
plt.plot(df3[['temp','hum']], label='Temperature')  # tempカラムをプロット
plt.title('Temperature over Time')  # グラフのタイトル
plt.xlabel('Time')  # x軸のラベル
plt.ylabel('Temperature')  # y軸のラベル
plt.legend()  # 凡例を表示
plt.show()  # グラフを表示

# 欠損値がある行のインデックスを表示
missing_values_index = df3['atemp'].isnull()# df3['atemp']で欠損値がある行のインデックスを確認
print(missing_values_index[missing_values_index].index)

#欠損値付近の折れ線グラフを作成
df3['atemp'].loc[200:220].plot(kind='line')
plt.show()

#欠損値を線形補完する
df3['atemp'] = df3['atemp'].astype(float)
df3['atemp'] = df3['atemp'].interpolate()
df3['atemp'].loc[200:220].plot(kind='line')
plt.show()

#欠損値を重回帰予測モデルで穴埋め
iris_df = pd.read_csv('iris.csv')
non_df = iris_df.dropna()#欠損値を含む行を削除
from sklearn.linear_model import LinearRegression
x = non_df.loc[:,'がく片幅':'花弁幅']
t = non_df['がく片長さ']
model = LinearRegression()
model.fit(x,t)

#欠損行の抜き出し
condition = iris_df['がく片長さ'].isnull()
non_data = iris_df.loc[condition]

#欠損値
try:
    x = non_data.loc[:,'がく片長さ':'花弁幅']
    pred = model.predict(x)
    iris_df.loc[condition, 'がく片長さ'] = pred

except Exception as e:
    print(f"エラー内容：{e}")

#自転車データでマハラノビス距離を計算
from sklearn.covariance import MinCovDet
df4 = df3.loc[:,'atemp':'windspeed']#全ての行とatemp列からwindspeedカラムの範囲を取得
df4 = df4.dropna()#欠損値を削除
mcd = MinCovDet(random_state=0,support_fraction=0.7)#共分散を計算
mcd.fit(df4)#共分散を計算
distance = mcd.mahalanobis(df4)
print(distance)

#箱ひげ図で外れ値を見つける
distance = pd.Series(distance)#シリーズに変換
distance.plot(kind='box')
plt.show()

#様々な基本統計量を調べる
tmp = distance.describe()
print(tmp)

#四分位範囲を用いた外れ値の判定
iqr = tmp['75%'] - tmp['25%']#IQR計算
jougen = 1.5 * (iqr) + tmp['75%']#上限
kagen = tmp['25%'] - 1.5 * (iqr)#下限

#上限と下限の条件をもとに、シリーズで条件検索
outliner = distance[(distance > jougen) | (distance < kagen)]
print(outliner)

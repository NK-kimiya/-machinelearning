import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Wholesale.csv')
print(df.head(3))

#欠損値の確認
print(df.isnull().sum())

#ChannelとRegiomを削除
df = df.drop(['Channel','Region'], axis=1)

#データを標準化する
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc_df = sc.fit_transform(df)
sc_df = pd.DataFrame(sc_df,columns=df.columns)

#モデルの作成
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3,random_state=0)

#モデルに学習させる
model.fit(sc_df)

#クラスタリング結果を確認
print(model.labels_)

#クラスタリングを結果を追加
sc_df['cluster'] = model.labels_
print(sc_df.head(2))

#クラスタごとに集計する
print(sc_df.groupby('cluster').mean())

#棒グラフで表示する
cluster_mean = sc_df.groupby('cluster').mean()

# クラスタごとの平均値を棒グラフで表示
cluster_mean.plot(kind='bar', figsize=(10, 7))
plt.title('Cluster Mean Values')
plt.ylabel('Mean Value')
plt.xlabel('Cluster')
plt.xticks(rotation=0)  # クラスタ名を水平に表示
plt.show()

#クラスタ数2～30でSSEを調べる
sse_list = []
for n in range(2,31):
    model = KMeans(n_clusters=n,random_state=0)
    model.fit(sc_df)
    sse = model.inertia_#SSEの計算
    sse_list.append(sse)

print(sse_list)

#折れ線グラフを描画する
se = pd.Series(sse_list)
num = range(2,31)
se.index = num
se.plot(kind = 'line')
plt.show()

#折れ線グラフからクラスタ数を5にしてファイルに書き出す
model = KMeans(n_clusters=5, random_state=0)
model.fit(sc_df)
sc_df['cluster'] = model.labels_
sc_df.to_csv('Clustered_Wholesale.csv',index=False)
#前処理
import pandas as pd
df = pd.read_csv('iris.csv')
print(df.head(3))

#種類別の値を確認
syurui = df['種類'].unique()
print(syurui[0])

#種類データの出現回数のカウント
print(df['種類'].value_counts())

#末尾3行を表示
print(df.tail(3))

#データに欠損値があるかを調べる
print(df.isnull())

#列単位で欠損値を確認
print(df.isnull().any(axis=0))

#各列の合計値を計算
print(df.sum())

#各列に欠損値がいくつあるかを集計
tmp = df.isnull()
print(tmp.sum())

#欠損値が1つでもある行を削除した結果をdf2に代入
df2 = df.dropna(how='any',axis=0)
print(df2.tail(3))

#欠損値の穴埋め
#花弁の長さに欠損値があれば、0に置き換える
df['花弁長さ'] = df['花弁長さ'].fillna(0)
print(df.tail(3))

#数値列の各平値を計算
numeric_df = df.select_dtypes(include=['number'])#数値型の列のみを選択
mean_values = numeric_df.mean()
print(mean_values)

#標準偏差の計算
print(df.std())





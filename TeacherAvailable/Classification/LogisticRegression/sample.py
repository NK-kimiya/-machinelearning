import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('iris.csv')
print(df.head(2))

#平均値による欠損値の穴埋め
try:
    numerical_df = df.select_dtypes(include=['float64', 'int64'])# 数値列のみを選択
    mean_values = numerical_df.mean()# 数値列の平均値を計算
    train2 = df.fillna(mean_values)
    print(train2.head())
except  Exception as e:
    # エラーが発生したときの処理
    print("ゼロ除算エラーが発生しました。" , e)   
finally:
    # エラーの有無に関わらず最後に実行するコード
    print("エラーチェックが完了しました。")
    
#特徴データと正解データに分割
x = train2.loc[:, :'花弁幅']
t = train2['種類']

#特徴量の標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
new = sc.fit_transform(x)

#訓練データと検証用データに分割
x_train,x_val, y_train, y_val = train_test_split(
    new, t, test_size=0.2, random_state=0
)

#ロジスティック回帰による学習
model = LogisticRegression(random_state=0,C=0.1,multi_class='auto',solver='lbfgs')#変数cは正規化項で値が小さいほど回帰式の係数を小さくしようとする

#正解率を確認する
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
print(model.score(x_val,y_val))

#係数を確認する
print(model.coef_)

#新規データで予測する
x_new = [[1,2,3,4]]
print(model.predict(x_new))

#確率の予測結果を確認する
print(model.predict_proba(x_new))


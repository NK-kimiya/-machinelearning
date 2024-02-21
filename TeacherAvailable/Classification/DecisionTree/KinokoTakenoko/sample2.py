#CSVファイルの読み込み
import pandas as pd
from sklearn import tree
import pickle
df = pd.read_csv('KvsT.csv')
print(df.head(3))

#特定の列の参照
#指定した列の参照
print(df['身長'])
#複数の列を一度に参照
col = ['身長','体重']
print(df[col])

#特徴量の列を参照して、xに代入
xcol = ['身長','体重','年代']
x = df[xcol]
print(x)

#正解データを参照して、tに代入
t = df['派閥']
print(t)

#モデルの準備(決定木)
model = tree.DecisionTreeClassifier(random_state=0)
#学習の実行
model.fit(x,t)

#身長170cm,体重70kg,年齢20代のデータ
taro = [[170,70,20]]
print(model.predict(taro))

#正解率の計算
print(model.score(x,t))

#モデルの保存
with open('KinokoTakenoko.pkl','wb') as f:
    pickle.dump(model,f)
    
#モデルを変数に読み込む
with open('KinokoTakenoko.pkl','rb') as f:
    model2 = pickle.load(f)

suzuki = [[180,75,30]]
print(model2.predict(suzuki))
#欠損値以外の前処理は省略
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

df = pd.read_csv('cinema.csv')
df.fillna(df.mean(), inplace=True)# 欠損値を各列の平均値で埋める
x = df.loc[:,'SNS1':'original']
t = df['sales']
# データを訓練、検証、テストセットに分割
X_train, X_temp, y_train, y_temp = train_test_split(x, t, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# モデルの訓練
model = LinearRegression()
model.fit(X_train, y_train)

# 検証データでモデルの性能を評価(平均二乗誤差・平均絶対誤差)
val_predictions = model.predict(X_val)
mse_val = mean_squared_error(y_val, val_predictions)
mae_val = mean_absolute_error(y_val, val_predictions)
print(f'Validation MSE: {mse_val}, MAE: {mae_val}')

# テストデータでモデルの最終性能を評価(平均二乗誤差・平均絶対誤差)
test_predictions = model.predict(X_test)
mse_test = mean_squared_error(y_test, test_predictions)
mae_test = mean_absolute_error(y_test, test_predictions)
print(f'Test MSE: {mse_test}, MAE: {mae_test}')

# RMSEを計算
rmse_val = np.sqrt(mse_val)
rmse_test = np.sqrt(mse_test)
print(f'Validation RMSE: {rmse_val}')
print(f'Test RMSE: {rmse_test}')
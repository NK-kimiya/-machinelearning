import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
df = pd.read_csv('Boston.csv')
print(df.head(2))

# 数値カラムの欠損値を平均値で埋める
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()), axis=0)

# ダミー変数化
df_dummy = pd.get_dummies(df)

# データの標準化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_dummy)

# PCAインスタンスの作成（全ての主成分を用意）
pca = PCA()

# 標準化したデータにPCAを適用し、モデルにfitさせる
pca.fit(df_scaled)
# 変換したデータ（主成分スコア）を取得
pca_df = pca.transform(df_scaled)

# 主成分スコアをDataFrameに変換
pca_result_df = pd.DataFrame(data=pca_df)

# 寄与率の表示
explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = np.cumsum(explained_variance)

# 結果の表示
print(pca_result_df.head())
print("Explained Variance per Component:", explained_variance)
print("Cumulative Explained Variance:", explained_variance_cumulative)

# 累積寄与率から必要な主成分の数を計算する
information_threshold = 0.9  # 情報のしきい値を90%と設定
components_required = np.where(explained_variance_cumulative >= information_threshold)[0][0] + 1

print(f"累積寄与率が{information_threshold*100}%を超えるのに必要な主成分の数: {components_required}")

# PCAインスタンスの作成（主成分数を8に設定）
pca2 = PCA(n_components=8)

# 標準化したデータにPCAを適用し、モデルにfitさせる
pca_result2 = pca2.fit_transform(df_scaled)

# 変換したデータ（主成分スコア）をDataFrameに変換
pca_result_df2 = pd.DataFrame(data=pca_result2, columns=[f'PC{i+1}' for i in range(8)])

# DataFrameをCSVファイルとして保存
pca_result_df2.to_csv('boston_pca.csv', index=False)


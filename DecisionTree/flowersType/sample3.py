from sklearn import tree
from sklearn.model_selection import train_test_split
#モデルの作成
model = tree.DecisionTreeClassifier(max_depth = 2,random_state=0)
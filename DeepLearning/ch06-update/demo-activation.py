import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000,100)
node_num = 100#各隠れ層のニューロンの数
hidden_layer_size = 5#隠れ層が5層
activations = {}#ここにアクティベーションの結果を格納


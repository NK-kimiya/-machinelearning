import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.minist import load_mnist
from TwoLayerNet import TwoLayerNet

'''
順伝播→逆伝播→パラメータの更新
'''

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#ハイパーパラメーターの設定
iters_num = 10000#更新回数
train_size = x_train.shape[0]#訓練データのサイズ
batch_size = 100#バッチサイズ
learning_rate = 0.1#学習率

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    #ランダムにバッチを計算
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #誤差逆伝播法を使って勾配を計算する
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 学習過程の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 性能評価
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
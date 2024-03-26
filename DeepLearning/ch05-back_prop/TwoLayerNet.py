import sys, os
import numpy as np
from ..common.layers import *
from ..common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        #重みの初期値
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        #バイアスの初期値
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        #ニューラルネットワークの層を順序付きで管理するための構造を初期化
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine = Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self,x):
        #レイヤーを順伝播させる
        for layer in self.layers.value():
            x = layer.forward(x)
        
        return x
    
    #交差エントロピー誤差の損失関数
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    #ニューラルネットワークの予測精度を計算
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1 : t = np.argmax(t,axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    #ューラルネットワークの各パラメータ（に対する損失関数の勾配
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        
        return grads
    
    #ニューラルネットワークの重みとバイアスに対する損失関数の勾配を誤差逆伝播法
    def gradient(self,x,t):
        #forward
        self.loss(x,t)
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine1'].dw
        grads['b2'] = self.layers['Affine1'].db
        
        return grads
        
    
    
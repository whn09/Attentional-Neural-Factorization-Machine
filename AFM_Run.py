# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:19:40 2018

@author: minjiang
"""

import tensorflow as tf
import LoadData as DATA
from AFM_Model import AFM
import numpy as np
from time import time

# 定义参数
Path = 'data/'
# Path = '../dataset/log_preprocess/20181128/'
Save_file = 'model/'
Epoch = 10  # default: 100
Batch_size = 128
Em_factor = 64
Attention_factor = 64
Keep_prob = '[0.7]'
Lr = 0.05
Lamda_em = 0.0
Lamda_attention = 15.0
Optimizer = 'AdagradOptimizer'
Verbose = 1
Bn = 1
Activation = 'relu'
Early_stop = 1
Attention = 1
Fields = 10
Decay = 0.99

# 读取数据
data = DATA.LoadData(Path)

activation_function = tf.nn.relu
if Activation == 'sigmoid':
    activation_function = tf.sigmoid
elif Activation == 'tanh':
    activation_function == tf.tanh
elif Activation == 'identity':
    activation_function = tf.identity

# 训练
t1 = time()
model = AFM(data.features_M, Em_factor, Attention_factor, Epoch, Batch_size, Lr,
            eval(Keep_prob), Optimizer, Bn, Activation, Verbose, Early_stop,
            Attention, Fields, Lamda_attention, Lamda_em, Decay, Save_file)
print('data.features_M:', data.features_M)
print('data.features:', len(data.features))
print('data.Train_data:', len(data.Train_data), len(data.Train_data['X']), len(data.Train_data['Y']))
print('data.Train_data:', data.Train_data['X'][:5], data.Train_data['Y'][:5])
print('data.Validation_data:', len(data.Validation_data), len(data.Validation_data['X']), len(data.Validation_data['Y']))
print('data.Validation_data:', data.Validation_data['X'][:5], data.Validation_data['Y'][:5])
print('data.Test_data:', len(data.Test_data), len(data.Test_data['X']), len(data.Test_data['Y']))
print('data.Test_data:', data.Test_data['X'][:5], data.Test_data['Y'][:5])
model.train(data.Train_data, data.Validation_data, data.Test_data)

# 找到使验证集误差最小的迭代次数
best_epoch = np.argmin(model.valid_loss)
print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
       % (best_epoch + 1, model.train_loss[best_epoch], model.valid_loss[best_epoch], model.test_loss[best_epoch],
          time() - t1))

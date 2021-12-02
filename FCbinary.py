"""经典的全连接模型用于测试 2分类和多分类"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os
# (1)指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0


def filter_01(x, y):
  keep = (y==0)|(y==1)
  x, y = x[keep], y[keep]
  y = y == 0
  return x,y


x_train, y_train = filter_01(x_train, y_train)
x_test, y_test = filter_01(x_test, y_test)


size_scale = 28
x_train_small = tf.image.resize(x_train, (size_scale, size_scale)).numpy()
x_test_small = tf.image.resize(x_test, (size_scale, size_scale)).numpy()

h_feat, w_feat, ch_size = x_test_small.shape[1], x_test_small.shape[2], x_test_small.shape[3]
# 定义模型(函数式)
inputs = layers.Input(shape=(h_feat, w_feat, ch_size))
x = layers.Flatten()(inputs)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(2, activation='softmax')(x)
model = Model(inputs=inputs, outputs=output)
# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train_small, y_train,
                    epochs=5,
                    batch_size=16,
                    validation_data=(x_test_small, y_test),
                    )

# 画图
plt.plot(history.history['val_accuracy'], label='Train_accuracy')
plt.show()

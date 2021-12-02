# 测试经典卷积神经网络

from tensorflow import keras
import tensorflow.keras.layers as layers
import os
# (1)指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1(cpu)

# 导入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# 数据处理，维度要一致，-1代表那个地方由其余几个值算来的
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32')
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32')
x_train = x_train / 255
x_test = x_test / 255

# 定义模型方法一-add()
# 这里我的模型比较简单，但准确率也有98%多，感兴趣的可以扩大深度或者引入其他方法优化下
model = keras.Sequential()  # 卷积--》池化--》平展化--》全连接--》全连接
model.add(layers.Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(3, 3),
                        strides=(1, 1), padding='valid',
                        activation='relu'))  # 卷积层加激活
model.add(layers.MaxPool2D(pool_size=(2, 2)))  # 池化层
model.add(layers.Flatten())  # 平展化
model.add(layers.Dense(32, activation='relu'))  # 全连接层
model.add(layers.Dense(10, activation='softmax'))  # 分类层

# 训练模型配置

model.compile(optimizer=keras.optimizers.Adam(),
              # 损失函数多分类使用交叉熵（这里还要看标签是否为one-hot编码），回归问题用均方差
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()  # 查看模型

# 进行模型训练
# history = model.fit(x_train, y_train,
#                     batch_size=64,
#                     epochs=5,
#                     validation_split=0.1)
# # 保存模型（这个方法比较常用，也可以考虑适合部署的SavedModel 方式）
# model.save('cnn1_save1.h5')#保存模型，名字可以任取，但要由.h5后缀
# #测试模型
# model.evaluate(x_test, y_test)

history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=16,
                    validation_data=(x_test, y_test),
                    validation_steps=200)

# 保存模型，名字可以任取，但要由.h5后缀
model.save('cnnTestDemo.h5')
# 验证模型
model.evaluate(x_test, y_test)
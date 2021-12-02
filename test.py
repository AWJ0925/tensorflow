from tensorflow import keras
import tensorflow.keras.layers as layers


# 导入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 数据处理，维度要一致，-1代表那个地方由其余几个值算来的
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255


# 定义模型方法一-add()
def myModel():
    # 这里我的模型比较简单，但准确率也有98%多，感兴趣的可以扩大深度或者引入其他方法优化下
    model = keras.Sequential()
    model.add(layers.Conv2D(input_shape=(28, 28, 1),
                            filters=32, kernel_size=(3, 3),
                            strides=(1, 1), padding='valid', activation='relu'))  # 卷积层加激活
    # 池化层
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    # 全连接层
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # 分类层

    return model

model = myModel()
# 训练模型配置
# loss=keras.losses.CategoricalCrossentropy(), 损失函数多分类使用交叉熵（这里还要看标签是否为one-hot编码）
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()  # 查看模型

history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_data=(x_test, y_test),
                    validation_steps=200)

# 保存模型，名字可以任取，但要由.h5后缀
model.save('cnn1_save1.h5')

model.load_weights('cnn1_save1.h5')
print("测试")
model.evaluate(x_test, y_test)
print("测试完成")


# 进行模型训练
# history = model.fit(x_train, y_train,
#                     batch_size=64,
#                     epochs=5,
#                     validation_split=0.1)
# # 保存模型（这个方法比较常用，也可以考虑适合部署的SavedModel 方式）
# model.save('cnn1_save1.h5')#保存模型，名字可以任取，但要由.h5后缀
# #测试模型
# model.evaluate(x_test, y_test)


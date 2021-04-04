import os
import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential()
# 增加Dense 网络，64个节点，使用relu作为激活函数
model.add(layers.Dense(64, activation='relu'))
# 
model.add(layers.Dense(32, activation='relu'))
# 增加 softmax layer 作为输出层
model.add(layers.Dense(16, activation='softmax'))

###usage
# # 创建sigmoid 为激活函数的隐层
# layers.Dense(64, activation='sigmoid')
# layers.Dense(64, activation=tf.sigmoid)
# # L1 正则
# layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
# # L2 正则
# layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
# # 正交初始化权重
# layers.Dense(64, kernel_initializer='orthogonal')
# # 初始化偏置向量
# layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

# Adam 优化器，分类交叉熵作为损失函数
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# optimizer，如 tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, or tf.train.GradientDescentOptimizer；
# loss，常见损失函数 mean square error (mse), categorical_crossentropy, 和 binary_crossentropy; 
# metrics: 训练指标，如 MAE,mean_squared_error,categorical_accuracy;

# 使用均方差作为损失函数
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# 使用交叉熵作为损失函数
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

# 迭代次数1000 次，batch_size 100
model.fit(data, labels, epochs=1000, batch_size=100)

# tensorflow提供tf.keras.Model.evaluate和tf.keras.Model.predict进行模型评估和模型预测，
# 对于NumPy和tf.data.Dateset这两种数据格式均支持。
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
# 模型评估
model.evaluate(data, labels, batch_size=32)
model.evaluate(dataset, steps=30)
# 模型预测
result = model.predict(data, batch_size=32)
print(result.shape)
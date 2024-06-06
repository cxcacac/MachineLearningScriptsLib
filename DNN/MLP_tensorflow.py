import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 使用tensorflow自带的工具加载MNIST手写数字集合
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

# 查看数据的维度和target的维度
print(mnist.train.images.shape)
print(mnist.train.labels.shape)


# 网格参数设置--三层网络结构
n_input = 784  # MNIST 数据输入(28*28*1=784)
n_hidden_1 = 512  # 第一个隐层
n_hidden_2 = 256  # 第二个隐层
n_hidden_3 = 128  # 第三个隐层
n_classes = 10  # MNIST 总共10个手写数字类别

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# 权重参数
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayerPerceptron(x, weights, biases):
    """
    # 前向传播 y = wx + b
    :param x: x
    :param weights: w
    :param biases: b
    :return:
    """
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
    layer2_drop = tf.nn.dropout(layer2, keep_prob)
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2_drop, weights['h3']), biases['b3']))
    # 计算第输出层。
    outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])

    return outLayer

predictValue = multilayerPerceptron(X, weights, biases)

learnRate = 0.01
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictValue, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(loss)

# 验证数据
correct_prediction = tf.equal(tf.argmax(predictValue, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print("FUNCTIONS READY!!")

init = tf.global_variables_initializer()

# 训练的总轮数
trainEpochs = 20
# 每一批训练的数据大小
batchSize = 128
# 信息显示的频数
displayStep = 5

with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 训
    for epoch in range(trainEpochs):
        avg_loss = 0.
        totalBatch = int(mnist.train.num_examples/batchSize)
        # 遍历所有batch
        for i in range(totalBatch):
            batchX, batchY = mnist.train.next_batch(batchSize)
            # 使用optimizer进行优化
            _, loss_value = sess.run([optimizer, loss], feed_dict={X: batchX, Y: batchY})
            # 求平均损失值
            avg_loss += loss_value/totalBatch

        # 显示信息
        if (epoch+1) % displayStep == 0:
            print("Epoch: %04d %04d Loss：%.9f" % (epoch, trainEpochs, avg_loss))
            train_acc = sess.run(accuracy, feed_dict={X: batchX, Y: batchY})
            print("Train Accuracy: %.3f" % train_acc)
            test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
            print("Test Accuracy: %.3f" % test_acc)
        
    print("Optimization Finished")
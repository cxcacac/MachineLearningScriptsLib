import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

fashion_mnist = input_data.read_data_sets('data/fashion/', one_hot=True)
print("No of images in training set {}".format(fashion_mnist.train.images.shape))
print("No of labels in training set {}".format(fashion_mnist.train.labels.shape))
print("No of images in test set {}".format(fashion_mnist.test.images.shape))
print("No of labels in test set {}".format(fashion_mnist.test.labels.shape))

labels = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'
}
# choose random one to show
img1 = fashion_mnist.train.images[41].reshape(28,28)
# Get corresponding integer label from one-hot encoded data
label1 = np.where(fashion_mnist.train.labels[41] == 1)[0][0]
# Plot sample
print("y = {} ({})".format(label1, labels[label1]))
plt.imshow(img1, cmap='Greys')

# constructions
# None means undetermined batch size.
x = tf.placeholder(tf.float32, [None, 784])
# set -1 when batch size undetermined, and it will be dynamically changed during training.
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# conv2d actually perform the convolutional operation.
# element wise multiplication of the input matrix (x) by the filter (w) with a stride of 1 and same padding. 
# The first and last values of strides are set to 1 which implies we don't want to move between training samples and different channels. 
# The second and third values of strides are also set to 1 which implies we move the filter by 1 pixel in both height and width direction.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#  We perform max pooling with stride of 2 and same padding, ksize implies our pooling window shape.
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# We will use a 5*5 filter and we set our filter size as 32. Since we use grayscale images, our input channel number will be 1. 
# 1 means input chanel number, 32 means output chanel number, same with 32 and 64.
w_c1 = tf.Variable(tf.random_normal([5,5,1,32]))
w_c2 = tf.Variable(tf.random_normal([5,5,32,64]))
b_c1 = tf.Variable(tf.random_normal([32]))
b_c2 = tf.Variable(tf.random_normal([64]))
# two convolutional layers.
conv1 = tf.nn.relu(conv2d(x_shaped, w_c1) + b_c1)
conv1 = maxpool2d(conv1)
conv2 = tf.nn.relu(conv2d(conv1, w_c2) + b_c2)
conv2 = maxpool2d(conv2)
# After two convolution layers with convolution and pooling operation, our input image will be downsampled from 28 * 28 * 1 to 7 * 7 *1. 
# We need to flatten this output before feeding to the fully connected layer. 
# Then we feed this flattened output to the fully connected layer and we multiply with weights and add bias and apply relu activations.
x_flattened = tf.reshape(conv2, [-1, 7*7*64])
w_fc = tf.Variable(tf.random_normal([7*7*64,1024]))
b_fc = tf.Variable(tf.random_normal([1024]))
fc = tf.nn.relu(tf.matmul(x_flattened,w_fc)+ b_fc)

w_out = tf.Variable(tf.random_normal([1024, 10]))
b_out = tf.Variable(tf.random_normal([10]))

output = tf.matmul(fc, w_out)+ b_out
yhat = tf.nn.softmax(output)

learning_rate = 0.01
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 10
batch_size = 100

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
   sess.run(init_op)
   total_batch = int(len(fashion_mnist.train.labels) / batch_size)
    # For each epcoh
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = fashion_mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =""{:.3f}".format(avg_cost))
        
   print(sess.run(accuracy, feed_dict={x:fashion_mnist.test.images, y:fashion_mnist .test.labels}))
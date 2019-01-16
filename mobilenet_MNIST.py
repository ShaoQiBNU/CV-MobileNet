##################### load packages #####################
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data


##################### load data ##########################
mnist = input_data.read_data_sets("mnist_sets", one_hot=True)

##################### set net hyperparameters #####################
learning_rate = 0.01

epochs = 200
batch_size_train = 128
batch_size_test = 100

display_step = 20

########### set net parameters ##########
#### img shape:28*28 ####
n_input = 784

#### 0-9 digits ####
n_classes = 10

##################### placeholder #####################
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


##################### build net model ##########################
def depthwise_separable_conv(inputs, filter, kernel_size, stride):
    layer1 = slim.separable_conv2d(inputs, num_outputs=None, stride=stride, depth_multiplier=1, kernel_size=kernel_size)
    layer1 = slim.batch_norm(layer1)
    layer1 = tf.nn.relu(layer1)

    layer2 = slim.conv2d(layer1, num_outputs=filter, stride=1, kernel_size=[1, 1], padding='SAME')
    layer2 = slim.batch_norm(layer2)
    layer2 = tf.nn.relu(layer2)

    return layer2


##################### MobileNet #####################
def MobileNet(x, n_classes):

    ####### reshape input picture ########
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    ###### first convolution ######
    conv1 = slim.conv2d(x, num_outputs=32, kernel_size=[3, 3], stride=2, padding='SAME')
    conv1 = slim.batch_norm(conv1)
    conv1 = tf.nn.relu(conv1)

    ###### depthwise separable convolution ######
    conv_dw1 = depthwise_separable_conv(conv1, filter=64, kernel_size=[3, 3], stride=1)

    conv_dw2 = depthwise_separable_conv(conv_dw1, filter=128, kernel_size=[3, 3], stride=2)

    conv_dw3 = depthwise_separable_conv(conv_dw2, filter=128, kernel_size=[3, 3], stride=1)

    conv_dw4 = depthwise_separable_conv(conv_dw3, filter=256, kernel_size=[3, 3], stride=2)

    conv_dw5 = depthwise_separable_conv(conv_dw4, filter=256, kernel_size=[3, 3], stride=1)

    conv_dw6 = depthwise_separable_conv(conv_dw5, filter=512, kernel_size=[3, 3], stride=2)

    conv_dw7 = depthwise_separable_conv(conv_dw6, filter=512, kernel_size=[3, 3], stride=1)
    conv_dw7 = depthwise_separable_conv(conv_dw7, filter=512, kernel_size=[3, 3], stride=1)
    conv_dw7 = depthwise_separable_conv(conv_dw7, filter=512, kernel_size=[3, 3], stride=1)
    conv_dw7 = depthwise_separable_conv(conv_dw7, filter=512, kernel_size=[3, 3], stride=1)
    conv_dw7 = depthwise_separable_conv(conv_dw7, filter=512, kernel_size=[3, 3], stride=1)

    conv_dw8 = depthwise_separable_conv(conv_dw7, filter=1024, kernel_size=[3, 3], stride=2)

    conv_dw9 = depthwise_separable_conv(conv_dw8, filter=1024, kernel_size=[3, 3], stride=1)

    ####### 全局平均池化 ########
    # pool1 = slim.avg_pool2d(conv_dw9, [7,7])
    pool1 = slim.avg_pool2d(conv_dw9, [1, 1], stride=1)

    ####### flatten 影像展平 ########
    flatten = tf.reshape(pool1, (-1, 1 * 1 * 1024))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out = tf.layers.dense(flatten, n_classes)

    return out


##################### define model, loss and optimizer #####################
#### model pred 影像判断结果 ####
pred = MobileNet(x, n_classes)

#### loss 损失计算 ####
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


##################### train and evaluate model ##########################
########## initialize variables ##########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    #### epoch 世代循环 ####
    for epoch in range(epochs + 1):

        #### iteration ####
        for _ in range(mnist.train.num_examples // batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y = mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples // batch_size_test):
        batch_x, batch_y = mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))

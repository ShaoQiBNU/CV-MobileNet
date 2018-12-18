#################### load packages ####################
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.examples.tutorials.mnist import input_data


################### load data ####################
mnist = input_data.read_data_sets("mnist_sets", one_hot=True)


################## set net hyperparameters ##################
learning_rate = 0.01

epochs = 20
batch_size_train = 128
batch_size_test = 100

display_step = 20

########## set net parameters ##########
#### img shape:28*28 ####
n_input = 784

#### 0-9 digits ####
n_classes = 10

###################### placeholder #######################
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


##################### build net model ##########################
def inverted_bottleneck(inputs, t, channel, n, stride):

    input = inputs

    for i in range(n):

        if i > 0:
            stride = 1

        out = tc.layers.conv2d(input, num_outputs=t*input.get_shape().as_list()[-1], kernel_size=1, stride=1,
                               normalizer_fn=tc.layers.batch_norm, activation_fn=tf.nn.relu6, padding='SAME')

        out = tc.layers.separable_conv2d(out, num_outputs=None, kernel_size=3, stride=stride,
                                         normalizer_fn=tc.layers.batch_norm, activation_fn=tf.nn.relu6, padding='SAME')

        out = tc.layers.conv2d(out, num_outputs=channel, kernel_size=1, stride=1,
                               normalizer_fn=tc.layers.batch_norm, activation_fn=None, padding='SAME')

        if stride == 1 and i > 0:
            out = tf.add(out, input)

        input = out

    return out


####################### MobileNetV2 #########################
def MobileNetV2(x, n_classes):

    ####### reshape input picture ########
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    ###### first convolution ######
    conv1 = tc.layers.conv2d(x, num_outputs=32, kernel_size=3, stride=2,
                             normalizer_fn=tc.layers.batch_norm, padding='SAME')

    ###### inverted bottleneck 1 ######
    invert_conv1 = inverted_bottleneck(conv1, t=1, channel=16, n=1, stride=1)

    ###### inverted bottleneck 2 ######
    invert_conv2 = inverted_bottleneck(invert_conv1, t=6, channel=24, n=2, stride=2)

    ###### inverted bottleneck 3 ######
    invert_conv3 = inverted_bottleneck(invert_conv2, t=6, channel=32, n=3, stride=2)

    ###### inverted bottleneck 4 ######
    invert_conv4 = inverted_bottleneck(invert_conv3, t=6, channel=64, n=4, stride=2)

    ###### inverted bottleneck 5 ######
    invert_conv5 = inverted_bottleneck(invert_conv4, t=6, channel=96, n=3, stride=1)

    ###### inverted bottleneck 6 ######
    invert_conv6 = inverted_bottleneck(invert_conv5, t=6, channel=160, n=3, stride=2)

    ###### inverted bottleneck 7 ######
    invert_conv7 = inverted_bottleneck(invert_conv6, t=6, channel=320, n=1, stride=1)

    ###### first convolution ######
    conv2 = tc.layers.conv2d(invert_conv7, num_outputs=1280, kernel_size=3, stride=1,
                             normalizer_fn=tc.layers.batch_norm, padding='SAME')

    ###### pool ######
    # pool1 = tc.layers.avg_pool2d(conv2, kernel_size=7)

    ####### flatten 影像展平 ########
    flatten = tf.reshape(conv2, (-1, 1 * 1 * 1280))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out = tf.layers.dense(flatten, n_classes)

    return out


###################### define model, loss and optimizer ####################
#### model pred 影像判断结果 ####
pred = MobileNetV2(x, n_classes)

#### loss 损失计算 ####
## 阻断label的梯度流 ##
y = tf.stop_gradient(y)
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
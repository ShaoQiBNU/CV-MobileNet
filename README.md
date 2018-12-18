MobileNets
==========

# 一. 背景介绍

> MobileNets是Google针对手机等嵌入式设备提出的一种轻量级的深层神经网络，是基于一个流线型的架构，它使用深度可分离的卷积来构建轻量级的深层神经网络，偏向于模型压缩方面，核心思想就是卷积核的巧妙分解，可以有效减少网络参数。与ImageNet分类上的其他流行的网络模型相比，MobileNets表现出很强的性能，并且广泛应用于物体检测，细粒度分类，人脸属性和大规模地理定位。

# 二. 网络详解

> MobileNets模型基于深度可分解的卷积，它可以将标准卷积分解成一个深度卷积和一个点卷积（1 × 1卷积核）。深度卷积将每个卷积核应用到每一个通道，而1 × 1卷积用来组合通道卷积的输出。后文证明，这种分解可以有效减少计算量，降低模型大小，下面图说明了标准卷积是如何进行分解的（假设stride=1，padding为same，则输出size不变）：

![image](https://github.com/ShaoQiBNU/MobileNet/blob/master/images/1.png)

## 1.标准卷积过程

>(a)是标准卷积过程，输入影像size为<img src="https://latex.codecogs.com/svg.latex?D_{f}\times&space;D_{f}\times&space;M" title="D_{f}\times D_{f}\times M" />，卷积核大小为<img src="https://latex.codecogs.com/svg.latex?D_{k}\times&space;D_{k}\times&space;M\times&space;N" title="D_{k}\times D_{k}\times M\times N" />，标准卷积过程计算如下：
>
>1. 1个卷积核与输入影像进行卷积运算
>
>   当输出影像大小为<img src="https://latex.codecogs.com/svg.latex?D_{f}\times&space;D_{f}" title="D_{f}\times D_{f}" />时，卷积计算量与输出影像大小有关，卷积核共作了<img src="https://latex.codecogs.com/svg.latex?D_{f}\times&space;D_{f}" title="D_{f}\times D_{f}" />次卷积运算，计算量为<img src="https://latex.codecogs.com/svg.latex?D_{f}\times&space;D_{f}" title="D_{f}\times D_{f}" />；
>
>   每次卷积运算的计算量与卷积核大小有关，计算量为<img src="https://latex.codecogs.com/svg.latex?D_{k}\times&space;D_{k}" title="D_{k}\times D_{k}" />，例如3x3的卷积核，每次卷积运算时要算9次乘法；
>
>   所以一个卷积核的计算量为<img src="https://latex.codecogs.com/svg.latex?D&space;_{f}\times&space;D&space;_{f}\times&space;D_{k}\times&space;D_{k}\times&space;M" title="D _{f}\times D _{f}\times D_{k}\times D_{k}\times M" />，<img src="https://latex.codecogs.com/svg.latex?M" title="M" />为输入影像的通道数。
>
>2. <img src="https://latex.codecogs.com/svg.latex?N" title="N" />个卷积核与影像进行卷积运算
>
>   <img src="https://latex.codecogs.com/svg.latex?N" title="N" />个卷积核与影像进行卷积运算的总计算量为<a href="https://www.codecogs.com/eqnedit.php?latex=D_{f}\times&space;D_{f}\times&space;D_{k}&space;\times&space;D_{k}\times&space;M\times&space;N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?D_{f}\times&space;D_{f}\times&space;D_{k}&space;\times&space;D_{k}\times&space;M\times&space;N" title="D_{f}\times D_{f}\times D_{k} \times D_{k}\times M\times N" /></a>

## 2. MobileNet卷积过程

>(b)(c)是MobileNet卷积过程，(b)是Depthwise Convolution过程，即逐通道的卷积；(c)是Pointwise过程，具体分解过程如下：
>
>1. (b)Depthwise
>
>   对于输入影像的每个通道分别用<a href="https://www.codecogs.com/eqnedit.php?latex=D_{k}&space;\times&space;D_{k}\times&space;1\times&space;1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?D_{k}&space;\times&space;D_{k}\times&space;1" title="D_{k} \times D_{k}\times 1" /></a>的卷积核进行卷积，共使用了<a href="https://www.codecogs.com/eqnedit.php?latex=M" target="_blank"><img src="https://latex.codecogs.com/svg.latex?M" title="M" /></a>个卷积核，从而得到<a href="https://www.codecogs.com/eqnedit.php?latex=M" target="_blank"><img src="https://latex.codecogs.com/svg.latex?M" title="M" /></a>个<a href="https://www.codecogs.com/eqnedit.php?latex=D_{f}&space;\times&space;D_{f}\times&space;1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?D_{f}&space;\times&space;D_{f}\times&space;1" title="D_{f} \times D_{f}\times 1" /></a>的特征图，这些特征图分别是从输入的不同通道学习而来，彼此独立，这一步的计算量为<a href="https://www.codecogs.com/eqnedit.php?latex=D_{f}\times&space;D_{f}\times&space;D_{k}&space;\times&space;D_{k}\times&space;M" target="_blank"><img src="https://latex.codecogs.com/svg.latex?D_{f}\times&space;D_{f}\times&space;D_{k}&space;\times&space;D_{k}\times&space;M" title="D_{f}\times D_{f}\times D_{k} \times D_{k}\times M" /></a>
>
>2. (c)Pointwise
>
>   对于上一步得到的<a href="https://www.codecogs.com/eqnedit.php?latex=M" target="_blank"><img src="https://latex.codecogs.com/svg.latex?M" title="M" /></a>个特征图作为<a href="https://www.codecogs.com/eqnedit.php?latex=M" target="_blank"><img src="https://latex.codecogs.com/svg.latex?M" title="M" /></a>个通道的输入，采用<a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?N" title="N" /></a>个<a href="https://www.codecogs.com/eqnedit.php?latex=1\times&space;1\times&space;M" target="_blank"><img src="https://latex.codecogs.com/svg.latex?1\times&space;1\times&space;M" title="1\times 1\times M" /></a>的卷积核进行标准卷积，得到<a href="https://www.codecogs.com/eqnedit.php?latex=D_{f}\times&space;D_{f}\times&space;M" target="_blank"><img src="https://latex.codecogs.com/svg.latex?D_{f}\times&space;D_{f}\times&space;M" title="D_{f}\times D_{f}\times M" /></a>的输出，这一步的计算量为<a href="https://www.codecogs.com/eqnedit.php?latex=D_{f}\times&space;D_{f}\times&space;M\times&space;N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?D_{f}\times&space;D_{f}\times&space;M\times&space;N" title="D_{f}\times D_{f}\times M\times N" /></a>

## 3. 计算量对比

> 将1和2的计算量进行对比，如下：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{D_{k}\times&space;D_{k}\times&space;M\times&space;D_{f}\times&space;D_{f}&space;&plus;&space;M\times&space;N\times&space;D_{f}\times&space;D_{f}}{D_{k}\times&space;D_{k}\times&space;M\times&space;N\times&space;D_{f}\times&space;D_{f}}=\frac{1}{N}&plus;\frac{1}{D_{k}^{2}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\frac{D_{k}\times&space;D_{k}\times&space;M\times&space;D_{f}\times&space;D_{f}&space;&plus;&space;M\times&space;N\times&space;D_{f}\times&space;D_{f}}{D_{k}\times&space;D_{k}\times&space;M\times&space;N\times&space;D_{f}\times&space;D_{f}}=\frac{1}{N}&plus;\frac{1}{D_{k}^{2}}" title="\frac{D_{k}\times D_{k}\times M\times D_{f}\times D_{f} + M\times N\times D_{f}\times D_{f}}{D_{k}\times D_{k}\times M\times N\times D_{f}\times D_{f}}=\frac{1}{N}+\frac{1}{D_{k}^{2}}" /></a>
>
> 当卷积核为3x3时，可以发现2的计算量能节约9倍左右。

# 三. 网络结构

> MobileNets Body Architecture由多个深度可分解卷积单元叠加而成，共1 + 2*13 + 1 = 28层。其采用的深度可分解卷积单元如图所示：

![image](https://github.com/ShaoQiBNU/MobileNet/blob/master/images/2.png)

> MobileNets的各层结构如下：

![image](https://github.com/ShaoQiBNU/MobileNet/blob/master/images/4.png)

# 四. 代码

> 利用MNIST数据集，构建MobileNets网络，查看网络效果，由于输入为28 x 28，所以最后的全局池化没有用到，代码如下：

```python
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

    ####### softmax ########
    out = tf.nn.softmax(out)

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
```

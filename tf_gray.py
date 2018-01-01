# -*- coding: utf-8 -*-
import tensorflow as tf
import math
import numpy as np
from PIL import Image
import os
import time

X_MAX = 1080.0
Y_MAX = 2160.0

def prepare_image(srcDir, dstDir):
    # 将图片处理为灰阶图并保存
    # 输入的文件名为 timestamp-x-y.png
    os.system('mkdir -p ' + dstDir)
    for srcFile in os.listdir(srcDir):
        name = srcFile[:-4]  # 去掉 .png
        timestamp, sx, sy = name.split('-')
        x = int(float(sx))
        y = int(float(sy))
        dstFile = '%s-%s-%s.png' % (timestamp, x, y)
        with Image.open(os.path.join(srcDir, srcFile)) as src:
            with src.convert('L') as dst:
                dst.save(os.path.join(dstDir, dstFile))
        pass
    pass


# prepare_image('/Users/ian/git/jump-bot/src/3', '/Users/ian/git/jump-bot/src/3-GRAY')
def prepare_data(start=0, end=0, srcDir='/Users/ian/git/jump-bot/src/3-GRAY'):
    files = [f for f in os.listdir(srcDir) if f[-4:] == '.png']
    end = len(files) if end == 0 else end
    fetchedFiles = files[start:end]
    return [parse_file_name(f, srcDir) for f in fetchedFiles]


def parse_file_name(file, dir):
    name = file[:-4]
    timestamp, sx, sy = name.split('-')
    return (int(sx) / X_MAX, int(sy) / Y_MAX, os.path.join(dir, file))


# print(prepare_data())
# data = prepare_data()
# xset = [d[0] for d in data]
# yset = [d[1] for d in data]
# minX = min(xset)
# maxX = max(xset)
# minY = min(yset)
# maxY = max(yset)
# print(minX, minY, maxX, maxY) # (234, 463, 892, 1069)
# 960 - 120 => 840
# 1360 - 360 => 1000
def load_input_from_image(file):
    with Image.open(file) as im:
        with im.crop((80, 700, 1000, 1500)) as im3:
            with im3.resize((im3.size[0] / 4, im3.size[1] / 4)) as im2:
                return np.array(im2).reshape(im2.size[0] * im2.size[1])


# print(load_input_from_image('/Users/ian/git/jump-bot/src/3-GRAY/1514604265.01-276-929.png'))

def model_func():
    batch_size = 100
    test_size = 20
    rankX = (920 / 4) * (800 / 4)
    rankY = 1
    rankW_H = rankX / 4

    data = prepare_data()
    # np.random.shuffle(data)
    # tr : test = 800 : 165
    trainData = data[:800]
    testData = data[800:]
    # trX = [load_input_from_image(d[2]) for d in trainData]
    # trY = [d[0] for d in trainData]
    print(len(trainData), len(testData))

    X = tf.placeholder('float', [None, rankX])
    Y = tf.placeholder('float', [])
    b = init_weights([rankY])
    # 初始化权重
    # w_h = init_weights([rankX, rankW_H])
    # w_h2 = init_weights([rankW_H, rankW_H])
    w_o = init_weights([rankX, rankY])

    # 生成网络模型，得到预测值
    py_x = model(X, w_o, b)
    # 定义损失函数
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = py_x, labels = Y))
    cost = -tf.reduce_sum(Y * tf.log(py_x))
    accuracy = tf.reduce_sum(tf.square(py_x - Y)) / batch_size
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(accuracy)
    # accuracy = tf.reduce_mean(tf.square(py_x - Y) / batch_size)
    predict_op = py_x

    # 训练及存储模型
    ckpt_dir = './ckpt_dir'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # 定义一个计数器，为训练轮数计数，不需要被训练
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # 在声明所有变量后调用tf.train.Saver
    saver = tf.train.Saver(tf.global_variables())
    # 位于saver之后的变量不会被存储

    # 训练模型并存储
    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        start = global_step.eval()  # 得到global_step的值
        print('Start from: ', start)

        for i in range(start, 1000):
            # 以 128 作为batch_size
            print('Trainning:', i)
            t0 = int(time.time() * 1000)
            for start, end in zip(range(0, len(trainData), batch_size),
                                  range(batch_size, len(trainData) + 1, batch_size)):
                t = trainData[start:end]
                trX = [load_input_from_image(d[2]) for d in t]
                trY = [[d[0]] for d in t]
                sess.run(train_op, feed_dict={X: trX, Y: trY})
                del trX
                del trY
                del t
            global_step.assign(i + 1).eval()  # 更新计数器
            t1 = int(time.time() * 1000)
            print('Saving at:', i, ' used:', (t1 - t0))
            saver.save(sess, ckpt_dir + '/model.ckpt', global_step=global_step)  # 存储模型
            np.random.shuffle(testData)
            test_data = testData[0:test_size]
            teX = [load_input_from_image(d[2]) for d in test_data]
            teY = [[d[0]] for d in test_data]
            print(i, sess.run(accuracy, feed_dict={X: teX, Y: teY}))
            print(teY[:1], sess.run(predict_op, feed_dict={X: teX[:1]}))


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_o, b):
    # 第一个全连接层
    # X = tf.nn.dropout(X, p_keep_input)
    # h = tf.nn.relu(tf.matmul(X, w_h))

    # h = tf.nn.dropout(h, p_keep_hidden)
    # 第二个全连接层
    # h2 = tf.nn.relu(tf.matmul(h, w_h2))
    # h2 = tf.nn.dropout(h2, p_keep_hidden)

    # 输出预测值
    return tf.matmul(X, w_o) + b


model_func()
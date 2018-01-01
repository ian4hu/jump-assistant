import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = np.array((
    (391, 590),
    (407, 620),
    (584, 845),
    (545, 755),
    (468, 685),
    (228, 325),
    (622, 855),
    (355, 470),
    (292, 390),
    (216, 330),
    (486, 740),
    (467, 675),
    (532, 750),
    (431, 650),
    (437, 650),
    (484, 735),
    (421, 640),
    (487, 750),
    (252, 340),
    (601, 845),
    (661, 920),
    (459, 670),
    (358, 460),
    (639, 865),
    (678, 930),
    (348, 430),
    (627, 855),
    (540, 755),
    (621, 845),
    (495, 745),
    (643, 855),
    (398, 610),
    (582, 840),
    (575, 755),
    (570, 810),
    (469, 680),
    (548, 760),
    (635, 855),
    (563, 760),
    (509, 750),
    (346, 460),
    (368, 460),
    (504, 750),
    (541, 755),
    (513, 750),
    (334, 420),
    (408, 620),
    (414, 625),
    (549, 760),
    (303, 465),
    (622, 850),
    (617, 820),
    (425, 600),
    (305, 460),
    (658, 860),
    (634, 840),
    (555, 750),
    (452, 630)
))

# Prepare train data
#train_X = np.linspace(250/700.0, 700/700.0, 100)
arr = np.array([d[0] for d in data])
scale = arr.max() * 1.0
train_X = arr * 1.0 / scale
#train_Y = 1.26 * train_X + np.random.randn(*train_X.shape) * (30 / 700.0) + (80/700.0)
train_Y = np.array([d[1] for d in data]) / scale

# Define the model
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - X*w - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    epoch = 1
    for i in range(1000):
        for (x, y) in zip(train_X, train_Y):
            _, w_value, b_value = sess.run([train_op, w, b],feed_dict={X: x,Y: y})
        print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value*700))
        epoch += 1


#draw
plt.plot(train_X,train_Y,"+")
plt.plot(train_X,train_X.dot(w_value)+b_value)
plt.show()
#!/usr/bin/python

import tensorflow as tf
import numpy as np

# tmp dir to store tmp data
TEMP_DIR = "./tmp"

# super parameters
learning_rate = 0.1
training_epoches = 100

# training data
train_X = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
train_Y = np.asarray([0.1, 1.2, 1.9, 2.8, 4.6, 4.9, 6.1, 6.9, 7.9, 9.0])

# training samples number
n_samples = train_X.shape[0]

# input placeholder
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# trainable model params
w = tf.Variable(np.random.randn(), name='weight')
tf.summary.scalar('weight', w)

b = tf.Variable(np.random.randn(), name='bias')
tf.summary.scalar('bias', b)

# simple linear model
pred = tf.add(tf.multiply(X, w), b)
tf.summary.scalar('pred', pred)

# cost function: mean square error
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / n_samples
tf.summary.scalar('cost', cost)

# optimizer: gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# create op to initialize variables
init = tf.global_variables_initializer()

# merge all summary op
merged_summary = tf.summary.merge_all()

# start train
with tf.Session() as sess:

    # run init
    sess.run(init)

    # create writer
    writer = tf.summary.FileWriter(TEMP_DIR, sess.graph)

    for epoch in xrange(training_epoches):
        for (x, y) in zip(train_X, train_Y):
            summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: x, Y: y})

        # save summary result into file each epoch
        summary = sess.run(merged_summary, feed_dict={X: train_X[0], Y: train_Y[0]})
        writer.add_summary(summary, epoch)

        c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Epoch:", "%04d" % (epoch + 1), "cost=%.9f" % (c), "w :", sess.run(w), "b :", sess.run(b)) 
    
    print("Optimization finished!")

#!/usr/bin/python

import tensorflow as tf

tf.reset_default_graph()

# Create variable
var1 = tf.get_variable("var1", shape=[1], initializer=tf.zeros_initializer)

# Use one operation to simulate training process
inc_op = var1.assign(var1 + 1)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    print "Before training: var1({})".format(sess.run(var1))

    sess.run(inc_op)

    print "After training: var1({})".format(sess.run(var1))

    # Save variables
    save_path = saver.save(sess, 'model/simple_network')
    print "model saved in file: model.ckpt"

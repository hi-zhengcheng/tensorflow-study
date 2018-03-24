#!/usr/bin/python

import tensorflow as tf

tf.reset_default_graph()

# Create variables. Since it will be restored, it do not need to be initialized.
var1 = tf.get_variable('var1', shape=[1])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'model/simple_network')
    print "model restored"

    # check value
    print "restored var1({})".format(sess.run(var1))

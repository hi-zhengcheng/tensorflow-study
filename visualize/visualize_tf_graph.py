#!/usr/bin/python

# This code snippet show how to summary tensorflow graph.
# And then, We can use tensorboard to visualize the graph.

import tensorflow as tf

# Temp dir to store temp data
TEMP_DIR = './tmp'

# simple constant add graph
const1 = tf.constant(3.0)
const2 = tf.constant(4.0, dtype=tf.float32)
const_add = const1 + const2

# simple placeholder add graph
holder1 = tf.placeholder(tf.float32, name='holder1')
holder2 = tf.placeholder(tf.float32, name='holder2')
holder_add = holder1 + holder2

# simple graph with name_scope
with tf.name_scope('hidden') as hidden_scope:
    const3 = tf.constant(5.0)
    const4 = tf.constant(6.0)
    const_add2 = const3 + const4

# combine all above small graphs
const5 = tf.constant(2.0)
all_graph = const_add + holder_add * const5 + const_add2

# save graph info into file
with tf.Session() as sess:
    # create writer 
    writer = tf.summary.FileWriter(TEMP_DIR, sess.graph)

    # must run the target graph within session
    sess.run(all_graph, feed_dict={holder1: 10, holder2: 20})


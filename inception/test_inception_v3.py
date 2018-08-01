from inception_v3 import inception_v3_base, inception_v3
from inception_utils import inception_arg_scope
import tensorflow as tf
slim = tf.contrib.slim
import os
import time

batch_size = 32
height, width = 299, 299
TMP_DIR = './tmp'
CHECKPOINT_PATH = './inception_v3.ckpt'
TEST_IMAGE_PATH = '/home/bing/project/im2txt_result/static/coco/train_val2014/COCO_val2014_000000537589.jpg'
GPU_ID = '5'

def main():
	# set gpu id
	os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
	
	# resnet input 
	X = tf.placeholder(tf.float32, [None, height, width, 3])

	# main resnet_101 network
	arg_scope = inception_arg_scope()
	with slim.arg_scope(arg_scope):
		net, end_points = inception_v3(X, 1001, is_training=False)
	net = tf.reshape(net, [-1, 1001])
	net = tf.argmax(net, 1)

	# image feature endpoint 
	mixed_7c_layer = end_points['Mixed_7c'] 
	mixed_7c_layer_shape = mixed_7c_layer.get_shape()
	global_pool_layer = slim.avg_pool2d(mixed_7c_layer, mixed_7c_layer_shape[1:3], padding="VALID", scope="mixed_7c_global_pool")
	image_feature = tf.squeeze(global_pool_layer, [1,2], name='image_feature_squeeze')
	image_feature = slim.flatten(global_pool_layer, scope='image_feature_flattern')

    # image feature endpoint (another method to compute image_feature)
    # mixed_7c_layer = end_points['Mixed_7c']
    # global_pool_layer = tf.reduce_mean(mixed_7c_layer, [1,2], keep_dims=True, name="mixed_7c_global_pool")
    # image_feature = tf.squeeze(global_pool_layer, [1,2], name='image_feature_squeeze')

	sess = tf.Session()

	# to visualize graph by tensorboard
	writer = tf.summary.FileWriter(TMP_DIR, sess.graph)

	# init variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# restore from checkpoint
	saver = tf.train.Saver(tf.global_variables())
	saver.restore(sess, CHECKPOINT_PATH)
	
	# prepare for input
	im = tf.read_file(TEST_IMAGE_PATH)
	im = tf.image.decode_jpeg(im)
	im = tf.image.resize_images(im, (height, width))
	im = tf.reshape(im, [-1, height, width, 3])
	im = tf.cast(im, tf.float32)
	inputs = im
	images = sess.run(inputs)

	# run network, get output and image_feature
	start_time = time.time()
	classId, feature = sess.run([net, image_feature], feed_dict={X:images})
	duration = time.time() - start_time

	print("run time: {}".format(duration))
	print("class id: {}".format(classId[0]))
	print("image feature shape:", feature[0].shape)

	sess.close()

if __name__ == '__main__':
	main()

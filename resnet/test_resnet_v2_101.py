from resnet_v2 import resnet_v2_101
from resnet_utils import resnet_arg_scope
import tensorflow as tf
slim = tf.contrib.slim
import os
import time

batch_size = 32
height, width = 224, 224

# you need to modify this params according to your env
TMP_DIR = './path_to_save_graph_file'
CHECKPOINT_PATH = '../resnet_v2_101.ckpt'
TEST_IMAGE_PATH = 'COCO_val2014_000000299448.jpg'
GPU_ID = '5'

def main():
	# set gpu id
	os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
	
	# resnet input 
	X = tf.placeholder(tf.float32, [None, height, width, 3])

	# main resnet_101 network
	arg_scope = resnet_arg_scope()
	with slim.arg_scope(arg_scope):
		net, end_points = resnet_v2_101(X, 1001, is_training=False)
	net = tf.reshape(net, [-1, 1001])
	net = tf.argmax(net, 1)

	# image feature endpoint 
	global_pool_layer = end_points['global_pool'] 
	image_feature = tf.squeeze(global_pool_layer, [1,2], name='image_feature_squeeze')

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
	print("image feature:")
	for i in range(len(feature[0])):
		print("{} : {}".format(i, feature[0][i]))

	sess.close()

if __name__ == '__main__':
	main()

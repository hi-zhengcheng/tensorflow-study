import tensorflow as tf
import os

def gen_image_process_net(image_path, is_training, output_path):
	# prepare image data
	img = tf.read_file(image_path)
	img = tf.image.decode_jpeg(img)

	# convert image values to [0, 1)
	img = tf.image.convert_image_dtype(img, dtype=tf.float32)
	
	# resize image 
	resize_height = 300
	resize_width = 300
	img = tf.image.resize_images(img, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR)

	# crop to target size
	target_height = 224
	target_width = 224
	if is_training:
		# random crop
		img = tf.random_crop(img, [target_height, target_width, 3])
		
		# random flip
		img = tf.image.random_flip_left_right(img)

		# distort colors
		img = tf.image.random_brightness(img, max_delta = 32. / 255.)
		img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
		img = tf.image.random_hue(img, max_delta=0.032)
		img = tf.image.random_contrast(img, lower=0.5, upper=1.5)

		# The random_* ops do not necessarily clamp.
		img = tf.clip_by_value(img, 0.0, 1.0)
		
	else:
		# entral crop/pad
		img = tf.image.resize_image_with_crop_or_pad(img, target_height, target_width)

	# convert image values to [0, 255]
	img = tf.image.convert_image_dtype(img, dtype=tf.uint8)

	# save file
	img = tf.image.encode_jpeg(img)
	net = tf.write_file(output_path, img)

	return net

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '5'
	test_image = 'path to test jpg image'
	net = gen_image_process_net(test_image, True, 'result.jpg')
	with tf.Session() as sess:
		sess.run(net)

if __name__ == '__main__':
	main()


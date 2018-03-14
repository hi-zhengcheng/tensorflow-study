#!/usr/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

def create_image_input_tensor(
    input_depth,
    target_with,
    target_height):
    """
    Args:
        image_file_path: string, absolute path for JPG image file.
        target_width: int, Desired image width for later process.
        target_height: int, Desired image height for later process.

    Returns:
        resized_image: tensor which will output the processed image.
        jpeg_data: placeholder for raw image data.
    """

    jpeg_data = tf.placeholder(tf.string, name='decode_jpeg_input')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)

    # The first dimension must be batch_size. Expand it to make batch_size is 1
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)

    # Just use tf.stack to convert a python list to a tensor, then call resize_bilinear
    resize_shape = tf.stack([target_height, target_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)

    return resized_image, jpeg_data

if __name__ == '__main__':
    # read image
    image_file_path = '/replace/to/your/own/test.jpg'
    image_data = gfile.FastGFile(image_file_path, 'rb').read()

    # create image input tensor
    target_width = 299
    target_height = 299
    input_depth = 3
    image_input_tensor, image_data_placeholder = create_image_input_tensor(input_depth, target_width, target_height)

    sess = tf.Session()

    # run the tensor. For feed_dict arg, you can also use: {'decode_jpeg_input:0': image_data}
    resized_image = sess.run(image_input_tensor, feed_dict={image_data_placeholder: image_data})

    # log result
    print "result shape: ", sess.run(tf.shape(resized_image))
    sess.run(resized_image)

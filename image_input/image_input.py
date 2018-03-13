#!/usr/bin/python

import numpy as np
import tensorflow as tf

def create_image_input_tensor(
    image_file_path,
    input_depth,
    target_with,
    target_height):
    """
    Args:
        image_file_path: string, absolute path for JPG image file.
        target_width: int, Desired image width for later process.
        target_height: int, Desired image height for later process.

    Return:
        Tensor which will output the processed image.
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

    return resized_image

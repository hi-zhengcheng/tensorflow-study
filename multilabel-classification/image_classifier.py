from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf

FLAGS = None

def create_model_graph(model_dir):
    """Create a graph from a pretrained saved model file

    Args:
        model_dir: dictionary where pretrained model is saved.

    Returns:
        graph: pretrained model graph
    """
    print("model_dir:", model_dir)


def main(_):
    # Set tensorflow log level
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create model graph from pretrained model.
    graph = create_model_graph(FLAGS.model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        type=str,
        default='',
        help='Path to image file.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='',
        help='Pretrained model dir.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

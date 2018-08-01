# Inception-v3

This example shows how to use pretrained Inception-v3 to do image classification and image feature extraction.

## Download
From [this page](https://github.com/tensorflow/models/tree/master/research/slim), download pretrained model and related network definition files.
* [pretrained model](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
* [inception_v3.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) , modify `from nets import inception_utils` to `import inception_utils`
* [inception_utils.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_utils.py)

## Attention
* `inception_v3.py` in `tensorflow/models` has some differents compared with the file in `tensorflow/tensorflow` git repository. Here I just use the network file in `tensorflow/models` repository. I have not test the network file in `tensorflow/tensorflow`.

## Step
1. In the test file, modify some params first.
1. `python test_inception_v3.py`
1. After running, use tensorboard to visualize the network graph.
1. In the test code, I just use `Mixed_7c` layer's output followed by a global average pooling as image feature.

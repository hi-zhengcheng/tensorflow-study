# Resnet-101

This example shows how to use pretrained resnet101 to do image classification and image feature extraction.

## Download
From [this page](https://github.com/tensorflow/models/tree/master/research/slim), download pretrained model and related network definition files.
* [pretrained model](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz)
* [resnet_v2.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) , modify `from nets import resnet_utils` to `import resnet_utils`
* [resnet_utils.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_utils.py)

## Attention
* **Latest** resnet_v2.py in `tensorflow/models` has some differents compared with the file in `tensorflow/tensorflow` git repository. Here I just use the network file in `tensorflow/models` repository. I have not test the network file in `tensorflow/tensorflow`.

## Step
1. In the test file, modify some params first.
1. `python test_resnet_v2_101.py`
1. After running, use tensorboard to visualize the network graph.

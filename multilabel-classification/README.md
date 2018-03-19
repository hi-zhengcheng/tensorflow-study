Multi Label Classification can be used in many scenes. In this post, I will study how it works in image classification. That is, give an image as input, then I can get which classes this image belongs to as output.

# 1 Singlelabel classification case study

First, we can study how single class classification works by reading [this post](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/tutorials/image_retraining.md#how-to-retrain-inceptions-final-layer-for-new-categories), and it's [source code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py).

By reading the post, we can know the main workflow is as follows:


1. Use Inception V3 model, we can obtain a 2048 dimension representation vector for an image.

1. Suppose now we have N image classes, we use a fully connected network to convert dimension of image representation vector from 2048 to N.

1. Use softmax, we can get N probability like values. For These N values, Imagine each index represent a class, and each value at a specific index represent the probability the image belongs to this class, then we can use this N valuse and the true image classes to create a cost function. For example, use cross-entropy.

1. Using gradient descent algorithms, we can update params in the fully connected network to make the cost function smaller.


[Inception V3 model detail](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)


# 2 Multilabel classification
The above post describes single label image classification and the code implementation only support sinlge label clasification. Actually, this method also support multilabel classificaion. And I will modify the code to make it work on multilable classification task.

I find [this repository](https://github.com/BartyzalRadek/Multi-label-Inception-net/blob/master/retrain.py) already implement the multilable classification.


## 2.1 repare training data
training data with multilabel

## 2.2 modify the code
modify the `retrain.py` to support multilabel classification.

## 2.3 test
test the trained model

Multi Label Classification can be used in many scenes. In this post, I will study how it works in image classification. That is, give an image as input, then I can get which classes this image belongs to as output.

Content

[TOC]

# Singlelabel classification case study

First, we can study how single class classification works by reading [this post](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/tutorials/image_retraining.md#how-to-retrain-inceptions-final-layer-for-new-categories), and it's [source code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py).

By reading the post, we can know the main workflow is as follows:


```flow
st=>start: start
io1=>inputoutput: raw image data
op1=>operation: image feature vector extraction model (Inception V3)
io2=>inputoutput: extracted image feature vector
op2=>operation: feature vector conversion (Fully connected network)
io3=>inputoutput: converted image feature vector with class count as dimension.
op3=>operation: softmax
io4=>inputoutput: multi classes probability distribution
e=>end: End

st->io1->op1->io2->op2->io3->op3->io4->e
```

1. Use Inception V3 model, we can obtain a 2048 dimension representation vector for an image.

1. Suppose now we have N image classes, we use a fully connected network to convert dimension of image representation vector from 2048 to N.

1. Use softmax, we can get N probability like values. For These N values, Imagine each index represent a class, and each value at a specific index represent the probability the image belongs to this class, then we can use this N valuse and the true image classes to create a cost function. For example, use cross-entropy.

1. Using gradient descent algorithms, we can update params in the fully connected network to make the cost function smaller.


[Inception V3 model detail](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)


# Multilabel classification
The above post describes single label image classification and the code implementation only support sinlge label clasification. Actually, this method also support multilabel classificaion. And I will modify the code to make it work on multilable classification task.

I find [this repository](https://github.com/BartyzalRadek/Multi-label-Inception-net/blob/master/retrain.py) already implement the multilable classification.


## prepare training data
training data with multilabel

## modify the code
modify the `retrain.py` to support multilabel classification.

## test
test the trained model

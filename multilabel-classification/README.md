# Multi label classification

Multi Label Classification can be used in many scenes. In this post, I will study how it works in image classification. That is, give an image as input, then I can get which classes this image belongs to as output. I find [this repository](https://github.com/BartyzalRadek/Multi-label-Inception-net/blob/master/retrain.py) already implement the multilable classification. Here I will implement one on scratch. 

## 1 Input data processing
1. Put all images in one folder.

1. Prepare `imglist_file`, `imglabel_file`:

    * imglist_file: txt file, each line contains two field seperated by one blank char. 
        ** field1: image_name
        ** filed2: how may labels this image belongs to

    * imglabel_file: txt file, each line contains ground truth labels corresponding to the image with same line number in imglist_file.


1. Create TFRecord files:

Find help by:

```
python create_tfrecord.py --h
```

1. Read data from TFRecord:

Find help by:

```
python read_tfrecord.py --h
```

Tensorflow uses queuing and threading to do high-performance training, especially in data reading. This [blog](http://adventuresinmachinelearning.com/introduction-tensorflow-queuing/) gives a clear tutorial to the tensorflow queuing and threading.And [this blog](http://machinelearninguru.com/deep_learning/data_preparation/tfrecord/tfrecord.html) gives a clear tutorial to tfrecord.

Keep in mind : 
* When writing tensorflow code, it actually builds up an **computation graph**. It doesn't run until you run it through `tf.Session` object.
* When runing **end node** of the **computation graph**, operation on the **root node** will first run, then go on, until reach to the **end node**.

## 2 Models
* Use Resnet, or inception model to extract image feature. We need to decide from which layer of these pretrained models, we can find a 'good' image feature.
* Add additional layers at the end
* ...

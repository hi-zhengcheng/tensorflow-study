# Image processing

In training task, if we do not have enough training images, we can do some image augmentation to increase training data set. Image augmentation can also make our network generalze well during the testing phase. This test script include following method to do image augmentation:

* resize
* random crop (during training phase)
* center crop/pad (during inference or test phase)
* random brightness
* random saturation
* random hue
* random contrast

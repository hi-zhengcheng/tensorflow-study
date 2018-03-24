# Save and Restore

When we handle a real world machine problem, the workflow is as flollows:

1. Define our model. Each model contains network, variables, constants and placeholders.

1. Train our model.

1. Save our model.

1. Restore model, and use it in product environment.

This post studies mainly on how to save and restore model data.

## 1 Save

Each model must have some trainable variables. The training step is actually to find the 'best values' for these variables. So When we find the best values, we can save them in files for later usage. Suppose we have a very simple model with one variable and one operations(we use one operation to simulate the training process). We can save variables like [this script](save_variable.py).

In the script, we just specify the model path by:
```
save_path = saver.save(sess, 'model/simple_network')
```

The `tf.train.Saver` will auto create model dir, and use *simple_network* as prefix of filenames of model files.

After running this script, one new `model` folder is created like this:

```
.
├── model
│   ├── checkpoint
│   ├── simple_network.data-00000-of-00001
│   ├── simple_network.index
│   └── simple_network.meta
└── save_variable.py
```
| File name | Describe |
| -- | -- |
| * .meta | A protocol buffer which saves the complete graph info.|
| *.data-00000-of-00001 <br/> *.index | These two binary files contains all the values of variables.|
| checkpoint | A plain text file which simply keeps a record of latest checkpoint files saved. |

The `meta` file just contain the graph structure info. It does not contain some actual values of *variables*. In training process. we often want to save one model version every n training iterations. To save the model at specific step number, We can use tf.train.Saver:

```python
saver = tf.train.Saver()
saver.save(sess, 'my_model', global_step=1000)
```

In fact, once we have saved the `meta` file the first time, we do not need to save `meta` file int the following time, because it does not change. We can do this as:
```python
saver.save(sess, 'my-model', global_step=step, write_meta_graph=False)
```

In the construction of `tf.train.Saver`, the API let you can control two things:
1. how many model versions you want to save in model dir at most.
1. The time duration to save the next model version.
```python
#saves a model every 2 hours and maximum 4 latest models are saved.
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
```
By using `max_to_keep`, you can call save freely without worry about your disk will be filled up by too many model versions. saver will automatically delete the old version model files. Pay special attention when use it with `write_meta_graph=False`. Consider the folloing code:
```python
import tensorflow as tf

a = tf.constant(1)
b = tf.Variable(tf.random_normal(shape=[2], name='var1'))

saver = tf.train.Saver(max_to_keep=2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver.save(sess, 'my-model', global_step=1, write_meta_graph=True)
saver.save(sess, 'my-model', global_step=2, write_meta_graph=False)
```
We let the saver save at most two versions model. When we call `save` API, only the firt time, we let the saver save the meta file. If you only call `save` two times, it's OK, and only one meta file was created in the model dir. Try call an additional `save`, the meta file created in the first step will be removed by the `max_to_keep=2` option.

## 2 Restore

When using someone else's pre-trained model, we have two things to do:
1. Create the graph network. We can write python program manually by defining every Variable, PlaceHolder and Operations. But if we have `meta` file, it can save a lot time for us:
    ```python
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    ```
    This will construct the model newtork in current graph. But the values of variables are not pre-trained values, we need to load values of variables manually.

1. Load the values. Values are saved this `*.index` and `*.data-00000-of-00001` file. We can load them as:
    ```python
    with tf.Session() as sess:
          new_saver = tf.train.import_meta_graph('my_model-1000.meta')
          new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    ```

Then, we can use `graph.get_tensor_by_name()` to get PlaceHolders, Variables, Constants and Operations. We can do inference or fine-tuning further more.




We can use following script to inspect specific tensor, or all tensors.
```
from tensorflow.python.tools import inspect_checkpoint as chkp
chkp.print_tensors_in_checkpoint_file('model/simple_network', tensor_name='var1', all_tensors=False)
chkp.print_tensors_in_checkpoint_file('model/simple_network', tensor_name='', all_tensors=True)
```


Thanks to [this good post](http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/) about save and restore. 

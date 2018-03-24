# Graph


## 1 Default graph
In tensorflow, every operation is added to one specific graph. If you do not define a graph manually, there is always one default graph. You can use following code to get the default graph:

```
tf.get_default_graph()
```

In some situations, we need to define multiple graphs and each graph contains some independent networks for different purposes. We can define graph manually as this:

## 2 Specify one graph
```
a = tf.constant(1)
my_graph = tf.Graph()
with my_graph.as_default():
    b = tf.constant(2)

c = tf.constant(3)
```
In above code, operations `consta 1` and `constant 3` are added to the default graph, `constant 2` is added to my_graph.

## 3 Information in graph

A graph contains two relevant kinds of information:

1. Graph Structure. It describes how model network is composed by operations.

1. Graph Collections. As the [official document](https://www.tensorflow.org/programmers_guide/graphs) writes:

>TensorFlow provides a general mechanism for storing collections of metadata in a `tf.Graph`. The `tf.add_to_collection` function enables you to associate a list of objects with a key (where `tf.GraphKeys` defines some of the standard keys), and `tf.get_collection` enables you to look up all objects associated with a key. Many parts of the TensorFlow library use this facility: for example, when you create a `tf.Variable`, it is added by default to collections representing "global variables" and "trainable variables". When you later come to create a `tf.train.Saver` or `tf.train.Optimizer`, the variables in these collections are used as the default arguments.

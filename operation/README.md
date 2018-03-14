# Operation

## 1 Basic concept
In tensorflow, there are things like constant, placeholder and variable. But actually they are all operations. See the following code:

```
tf.reset_default_graph()
const1 = tf.constant(1)
print tf.get_default_graph().as_graph_def()
```
The output is:

```
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
versions {
  producer: 22
}
```

We create a constant. But unlike other program languages(python, java, c++), We are not just allocate a block of memory and make a pointer point this memory block, what we actually do is create a `node` which is a `Const` `op`. We usually call it `op` short for `operation`. And the const value `1` we assigned to this `op` is actually the output of this `op`.

Besides of constant, variable and placeholder, `tf.add`, `tf.subtract` this kind of math operations are also `op`.

Additionally, all data process operation in tensorflw are all `op`.

Test following code and see the output:

```
tf.reset_default_graph()
const1 = tf.constant(1)
const2 = tf.constant(2)
result = tf.add(const1, const2)
print tf.get_default_graph().as_graph_def()
```

Output:

```
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "Add"
  op: "Add"
  input: "Const"
  input: "Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
versions {
  producer: 22
}
```

We define a tiny network only doing add operation. From the output, we can see, this tiny network consists of three `op`(two `Const`, one `Add`). The two `Const` op is the input of the `Add` op.

## 2 Tensor: output of operation

In tensorflow, we call the output of `op` as `tensor`. Some `op` defination will directly return one tensor for us, for example:

```
const1 = tf.constant(1)
placeholder1 = tf.placeholder(tf.float32)
var1 = tf.Variable(0)
```

An `op` will allocate memory for its outputs(tensors), which are available on endpoint like `nodename:0`. For example, if we have a tensor and we want to fetch the actually value, there are two methods availabe. See folloing code:

```
tf.reset_default_graph()

# define a constant op, and it return a tensor
a = tf.constant(3, name='const_op')
sess = tf.Session()

# method 1: use endpoint string to fetch value in tensor
sess.run('const_op:0')

# method 2: use tensor directly to fetch value in tensor
sess.run(a)
```

## 3 Feed placehoder tensor

See following code:

```
tf.reset_default_graph()
placeholder1 = tf.placeholder(tf.int32, name='test_placeholder)
placeholder2 = tf.placeholder(tf.int32, name='test_placeholder2')
add_result = tf.add(placeholder1, placeholder2, name='test_add')
sess = tf.Session()


sess.run('test_add:0', feed_dict={'test_placeholder:0': 1, placeholder2:2})
```
in feed_dict, we can also use the above two methods to pass value.

## 4 Conclusion
* In tensorflow, the most baisic thing is `op`. The graph consists of `op` nodes connected to each other.
* Tensor is the output of `op`.
* There are two methods to fetch or feed tensor:
    1. by tensor directly
    1. by endpoint string `nodename:0`

# Image to text

## 1 Basic concepts

### 1.1 cross entropy cost
In the evaluate step, im2txt model uses cross entropy as cost function:
```
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
```
* targets : the tensor with shape [-1], and each element is the specific word index in vocab.
* logits : the model output with shape[-1, padded_length], and each row is the word probability distribution.
* losses : has the same shape with targets tensor.

### 1.2 Perplexity
* Definition:
You can find many versions of perplexity definition after searching on Google. In the im2txt model, however, it uses the definition described in this [Quora post](https://www.quora.com/In-NLP-why-do-we-use-perplexity-instead-of-the-loss), and thanks for the writer's detailed explanation. Sum up, im2txt model uses this perplexity version: <div>**e**<sup>per-word-entropy-loss</div>

* From the above definition, we can see this version of perplexity is closely related to the cross entropy loss. When our training is go on, cross entropy loss goes smaller, perplexity too. So `cross entropy lose` and `perplexity` are equal as a measure of language model performance.

## 2 model details

### 2.1 image embedding(Inception V3 model)

### 2.2 word embedding

### 2.3 LSTM

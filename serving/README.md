# Tensorflow Serving

It is mainly for :

1. production environments.
1. Make it easy to deploy new algorithms and experiments.
1. Keeping the same server architecture and APIs.

It has following main concepts:

## Servables
* Servabes are the underlying objects that clients use to perform computation(for example, l lookup or inference)
* A servable can be: a lookup table, a single model, a tuple of inference models.
* A single server instance can load more than one version of servable concurrently.
* Clients may request the latest version or a specific version.

So, for a pretrained model, you can represent it as either of the following:
1. multiple independent servables. A servable may correspond to a fraction of a model.
1. a single servable.

## Loaders

it is mainly for:
1. manage a servable's life cycle.
1. standardize the APIs for loading and unloading a servable

## Sources

It is mainly for:
1. Create loaders for servable versions.
1. Communicate with manager by `Aspired Versions`, which represents the set of servable versions that should be loaded and ready

## Managers

handle the full lifecycle of Servables, including:
1. Communicate with source with `aspired version`
1. Applies the configured Version Policy to determine what to do: unload a previously loaded version, or load a new version
1. Clients communicate with mananger to request a handle to a version of servable.

---

The main workflow:

**Client** ---> **Manager** <--> **Source** <--> **Loader** <--> **Servable**

One pic from official document:
![sering_architecture](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/images/serving_architecture.svg)

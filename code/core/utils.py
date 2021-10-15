import os
import sys
from math import ceil
from datetime import datetime

import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid", {'axes.grid' : False})


def normalize(x, reduce_min=True, reduce_max=True):
  if reduce_min: x -= tf.reduce_min(x, axis=(-3, -2), keepdims=True)
  if reduce_max: x = tf.math.divide_no_nan(x, tf.reduce_max(x, axis=(-3, -2), keepdims=True))

  return x


def opt_or_default(options):
  def _get_options(param):
    return options.get(param, {})
  
  return _get_options


def visualize(
    image,
    title=None,
    rows=2,
    cols=None,
    figsize=(16, 7.2),
    cmap=None
):
  if image is not None:
    if isinstance(image, (list, tuple)) or len(image.shape) > 3:  # many images
      plt.figure(figsize=figsize)
      cols = cols or ceil(len(image) / rows)
      for ix in range(len(image)):
        plt.subplot(rows, cols, ix+1)
        visualize(image[ix],
                 cmap=cmap,
                 title=title[ix] if title is not None and len(title) > ix else None)
      plt.tight_layout()
      return

    if isinstance(image, tf.Tensor): image = image.numpy()
    if image.shape[-1] == 1: image = image[..., 0]
    plt.imshow(image, cmap=cmap)
  
  if title is not None: plt.title(title)
  plt.axis('off')


def unfreeze_top_layers(
    model: tf.keras.Model,
    layers: float,
    freeze_bn: bool
):
  if not layers:
    model.trainable = False
    return

  model.trainable = True

  frozen_layer_ix = int((1-layers) * len(model.layers))
  for ix, l in enumerate(model.layers):
    l.trainable = (ix > frozen_layer_ix and
                   (not isinstance(l, tf.keras.layers.BatchNormalization) or
                    not freeze_bn))
  
  print(f'Unfreezing {layers:.0%} layers. Bottom-most is the {frozen_layer_ix}-nth layer ({model.layers[frozen_layer_ix].name}).')


def get_optimizer(name, learning_rate):
  if name == 'adam': return tf.optimizers.Adam(learning_rate=learning_rate)
  if name == 'sgd': return tf.optimizers.SGD(learning_rate=learning_rate)
  if name == 'momentum': return tf.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
  
  return tf.optimizers.get({'class_name': name, 'config': {'learning_rate': learning_rate}})



def log_func_start(fun_name, *args, **kwargs):
  print(f'[{fun_name} started at {datetime.now()}]')

  max_param_size = max(list(map(len, kwargs.keys())) or [0])
  
  for v in args:
    print(f'  {v}')
  for k, v in kwargs.items():
    print(f'  {k:<{max_param_size}} = {v}')
  print()


def log_fn(fn):
  def _log_wrapper(*args, **kwargs):
    log_func_start(fn.__name__, *args, **kwargs)
    return fn(*args, **kwargs)

  return _log_wrapper



def get_preprocess_fn(preprocess_fn):
  if not isinstance(preprocess_fn, str):
    return preprocess_fn
  
  *mod, fn_name = preprocess_fn.split('.')
  mod = '.'.join(mod)

  if mod:
    return getattr(sys.modules[mod], fn_name)

  return globals[fn_name]

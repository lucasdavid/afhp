import argparse
import os
import sys
from datetime import datetime
from math import ceil
from typing import Union

import tensorflow as tf


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_extractor_params(args):
  TAG = args.strategy
  SEED = args.backbone_valid_seed
  B_TRAIN = args.batch_size
  S = args.patch_size
  P = args.patches_train
  POS = args.positives_train
  W = args.backbone_train_workers
  BACKBONE = args.backbone_architecture
  EPOCHS_HE = args.backbone_train_epochs
  EPOCHS_FT = args.backbone_finetune_epochs
  TARGET = "painter"
  if EPOCHS_HE + EPOCHS_FT == 0:
    NAME = f"pbn_{args.data_split}_{args.patch_size}_{BACKBONE}_{TAG}_pretrained"
  else:
    NAME = f"pbn_{args.data_split}_{args.patch_size}_{BACKBONE}_{TAG}_{EPOCHS_HE}_{EPOCHS_FT}"
  WEIGHTS = os.path.join(args.weights_dir, NAME + ".h5")
  WEIGHTS_EXIST = os.path.exists(WEIGHTS)

  return SEED, B_TRAIN, S, P, POS, W, BACKBONE, EPOCHS_HE, EPOCHS_FT, TARGET, NAME, WEIGHTS, WEIGHTS_EXIST


def get_afhp_params(args):
  STRATEGY = args.strategy
  BACKBONE = args.backbone_architecture
  EPOCHS_HE = args.backbone_train_epochs
  EPOCHS_FT = args.backbone_finetune_epochs

  PROBLEM = args.afhp_problem
  EPOCHS = args.afhp_epochs
  LR = args.afhp_lr
  D_STEPS = 5

  NOISE = 512
  FEATURES = 2048
  TASK_MODE = 'random'

  BACKBONE_NAME = f"{BACKBONE}_{STRATEGY}_{EPOCHS_HE}_{EPOCHS_FT}"
  NAME = f"pbn_{args.data_split}_{args.patch_size}_afhp_{EPOCHS}_ac{args.afhp_ac_w}_cf{args.afhp_cf_w}_gp{args.afhp_gp_w}-{BACKBONE_NAME}"
  WEIGHTS = os.path.join(args.weights_dir, NAME + ".h5")
  WEIGHTS_EXIST = os.path.exists(WEIGHTS)

  return PROBLEM, EPOCHS, LR, D_STEPS, NOISE, FEATURES, TASK_MODE, NAME, WEIGHTS, WEIGHTS_EXIST


def visualize(
    image,
    title=None,
    rows=2,
    cols=None,
    figsize=(16, 7.2),
    cmap=None
):
  import matplotlib.pyplot as plt

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
    layers: Union[float, str],
    freeze_bn: bool
):
  if not layers:
    model.trainable = False
    return

  model.trainable = True

  if layers == "all":
    t_idx = 0
  else:
    if isinstance(layers, str):
      t_idx = model.layers.index(model.get_layer(layers))
    else:
      t_idx = int((1-layers) * len(model.layers))

    for ix, l in enumerate(model.layers):
      l.trainable = (ix >= t_idx
                     and not (isinstance(l, tf.keras.layers.BatchNormalization)
                              and freeze_bn))

  total = len(model.layers)
  trainable = total - t_idx
  print(f'Trainable layers: {trainable} ({trainable / total:.0%}). '
        f'First is {model.layers[t_idx].name} ({t_idx}-nth layer).')


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


def log_calls(fn):
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

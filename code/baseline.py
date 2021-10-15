"""Painter by Numbers Baseline.

Strategy Description:

  - InceptionV3
  - Pyramid Features
  - Supervised Contrastive Learning
"""

# region Setup

import os
import tensorflow as tf

from core.boot import *

gpus_with_memory_growth()
dst = appropriate_distributed_strategy()

import os

import numpy as np

from core.callbacks import *
from core.metrics import *
from core.models.supcon import *
from core.training import *
from core.utils import *
from core.datasets import pbn

# endregion

# region Experiment Configuration

base_dir = os.environ.get('SCRATCH', '/scratch/lerdl/lucas.david')

data_dir = os.environ.get('DATA_DIR', os.path.join(base_dir, 'datasets', 'painter-by-numbers'))
logs_dir = os.environ.get('LOGS_DIR', os.path.join(base_dir, 'logs', 'painter-by-numbers'))
weights_dir = os.environ.get('WEIGHTS_DIR', os.path.join(base_dir, 'weights', 'painter-by-numbers'))

class ExperimentConfig:
  base_dir = base_dir

  np_seed = 21416
  tf_seed = 4769

class DataConfig:
  all_info   = f'{data_dir}/meta/all_data_info.csv'
  train_info = f'{data_dir}/meta/train_info.csv'
  train_records = f'{data_dir}/train.tfrecords'
  test_records  = f'{data_dir}/test.tfrecords'

  size                 = (299, 299)
  shape                = (*size, 3)
  batch_size           = 64
  patches              = 20
  shuffle_buffer_size  = 48 * batch_size
  prefetch_buffer_size = tf.data.experimental.AUTOTUNE
  shuffle_seed         = 2142

  valid_size = 0.3
  shards = 32

  preprocess = tf.keras.applications.inception_v3.preprocess_input
  deprocess = lambda x: (x + 1) * 127.5
  to_image = lambda x: tf.cast(tf.clip_by_value(DataConfig.deprocess(x), 0, 255), tf.uint8)

  class aug:
    brightness_delta =  .2
    saturation_lower =  .5
    saturation_upper = 1.0
    contrast_lower   =  .5
    contrast_upper   = 1.5
    hue_delta        =  .0

  R = tf.random.Generator.from_seed(ExperimentConfig.tf_seed, alg='philox')
  np.random.seed(ExperimentConfig.np_seed)

class SupconConfig:
  override = True
  input_tensor = tf.keras.Input(shape=DataConfig.shape, name='image')
  
  class model:
    backbone = tf.keras.applications.InceptionV3
    pooling = 'avg'
    pr_ks = {
      # kernel_regularizer=tf.keras.regularizers.l2(0.01)
    }

  class training:
    epochs = 2
    temperature = 0.05
    
    train_steps = None
    valid_steps = None
    initial_epoch = 0
    
    logs = os.path.join(logs_dir, 'i3-supcon-baseline')
    weights = os.path.join(weights_dir, 'i3-supcon-baseline')

    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9, nesterov=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_weights = {'artist': 0.4, 'style': 0.3, 'genre': 0.3}
    loss = {
      'artist': SupervisedContrastiveLoss(temperature),
      'style': SupervisedContrastiveLoss(temperature),
      'genre': SupervisedContrastiveLoss(temperature),
    }

    callbacks = [
      tf.keras.callbacks.TerminateOnNaN(),
      tf.keras.callbacks.EarlyStopping(patience=40, verbose=1),
      tf.keras.callbacks.TensorBoard(logs, histogram_freq=1, write_graph=True),
      tf.keras.callbacks.ModelCheckpoint(weights, save_weights_only=True, save_best_only=True, verbose=1),
      ReduceLRBacktrack(min_lr=1e-6, best_path=weights, distributed_strategy=dst),
    ]

  class finetune:
    epochs = 2
    layers = 0.6  # 60%
    freeze_bn = True

    train_steps = None
    valid_steps = None
    initial_epoch = 0
    
    logs = os.path.join(logs_dir, 'i3-supcon-baseline-ft')
    weights = os.path.join(weights_dir, 'i3-supcon-baseline-ft')
    exported = os.path.join(weights_dir, 'i3-supcon-baseline-ex')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    callbacks = [
      tf.keras.callbacks.TerminateOnNaN(),
      tf.keras.callbacks.EarlyStopping(patience=40, verbose=1),
      tf.keras.callbacks.TensorBoard(logs, histogram_freq=1, write_graph=True),
      tf.keras.callbacks.ModelCheckpoint(weights, save_weights_only=True, save_best_only=True, verbose=1),
      ReduceLRBacktrack(min_lr=1e-6, best_path=weights, distributed_strategy=dst),
    ]

# endregion

# region Painter by Numbers Dataset
info = pbn.PainterByNumbers.load_info(DataConfig.all_info, DataConfig.train_info)

dataset = tf.data.TFRecordDataset(DataConfig.train_records)
dataset = dataset.apply(tf.data.experimental.assert_cardinality(pbn.PainterByNumbers.num_train_samples))

shards_t = int((1 - DataConfig.valid_size) * DataConfig.shards)
shards_v = DataConfig.shards - shards_t

train = pbn.prepare(dataset, shards_t, augment=True, shuffle_seed=DataConfig.shuffle_seed, config=DataConfig)
valid = pbn.prepare(dataset, shards_v, shards_t, shuffle_seed=DataConfig.shuffle_seed + 429)

print('\nPainter by Numbers Stats')
print(f'  training samples: {pbn.PainterByNumbers.num_train_samples:10d}')
print(f'  testing  samples: {pbn.PainterByNumbers.num_train_samples:10d}')
print('  total shards:', DataConfig.shards)
print('  training shards:', shards_t)
print('  validation shards:', shards_v)

print('  training cardinality: ', dataset.cardinality().numpy())
print('  sub-training cardinality: ', train.cardinality().numpy())
print('  validation   cardinality: ', valid.cardinality().numpy())
print(f'  training   dataset: {train}')
print(f'  validation dataset: {train}')
# print('  testing  cardinality: ', dataset.cardinality().numpy())
print()

# endregion

# region Supervised Contrastive

### Network
with dst.scope():
  print(f'Loading {SupconConfig.model.backbone.__name__}')

  backbone = SupconConfig.model.backbone(
    classifier_activation=None,
    include_top=False,
    input_tensor=SupconConfig.input_tensor
  )
  
  backbone.training = False

  nn = supcon_encoder_multitask_network(
    input_tensor=SupconConfig.input_tensor,
    backbone=backbone,
    pooling=SupconConfig.model.pooling,
    artist_projection_head=ProjectionHead(name='artist_project_head', pr_ks=SupconConfig.model.pr_ks),
    style_projection_head=ProjectionHead(name='style_project_head', pr_ks=SupconConfig.model.pr_ks),
    genre_projection_head=ProjectionHead(name='genre_project_head', pr_ks=SupconConfig.model.pr_ks),
    name='painter-by-numbers/supcon'
  )

  nn.compile(
    optimizer=SupconConfig.training.optimizer,
    loss=SupconConfig.training.loss,
    loss_weights=SupconConfig.training.loss_weights
  )

### Training
# nn(tf.random.normal((DataConfig.batch_size, 299, 299, 3)), training=False);  # sanity check.

report = train_fn(
  nn,
  train,
  valid,
  epochs=SupconConfig.training.epochs,
  logs=SupconConfig.training.logs,
  weights=SupconConfig.training.weights,
  train_steps=SupconConfig.training.train_steps,
  valid_steps=SupconConfig.training.valid_steps,
  initial_epoch=SupconConfig.training.initial_epoch,
  callbacks=SupconConfig.training.callbacks,
  override=SupconConfig.override,
  distributed_strategy=dst,
)


### Fine-Tuning

initial_epoch = SupconConfig.finetune.initial_epoch or len(report['epochs'])

with dst.scope():
  unfreeze_top_layers(
  nn,
  SupconConfig.finetune.layers,
  SupconConfig.finetune.freeze_bn,
  )

  nn.compile(
    optimizer=SupconConfig.finetune.optimizer,
    loss=SupconConfig.training.loss,
    loss_weights=SupconConfig.training.loss_weights
  )

train_fn(
  nn,
  train,
  valid,
  epochs=SupconConfig.finetune.epochs,
  logs=SupconConfig.finetune.logs,
  weights=SupconConfig.finetune.weights,
  train_steps=SupconConfig.finetune.train_steps,
  valid_steps=SupconConfig.finetune.valid_steps,
  initial_epoch=initial_epoch,
  callbacks=SupconConfig.finetune.callbacks,
  override=SupconConfig.override,
  distributed_strategy=dst,
)

# endregion

print(f'Exporting model to {SupconConfig.finetune.exported}')
nn.save(SupconConfig.finetune.exported, save_format='tf')

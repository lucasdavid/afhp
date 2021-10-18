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
  patches                = int(os.environ.get('PATCHES', '20'))
  batch_size_per_replica = int(os.environ.get('BATCH', '128'))
  batch_size           = 128 * dst.num_replicas_in_sync
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
    perform = os.environ.get('PERFORM_T', 'true') == 'true'

    epochs = int(os.environ.get('EPOCHS', '100'))
    temperature = 0.05
    
    train_steps = None
    valid_steps = None
    initial_epoch = int(os.environ.get('INITIAL_EPOCH', '0'))
    
    learning_rate = float(os.environ.get('LR', '0.001'))

    optimizer_name = os.environ.get('OPT', 'momentum')
    optimizer = get_optimizer(optimizer_name, learning_rate)

    logs = os.path.join(logs_dir, f'i3-supcon-baseline-opt:{optimizer_name}-lr:{learning_rate}')
    weights = os.path.join(weights_dir, f'i3-supcon-baseline-opt:{optimizer_name}-lr:{learning_rate}')
    
    loss_weights = {'artist': 0.4, 'style': 0.3, 'genre': 0.3}
    loss = {
      'artist': SupervisedContrastiveLoss(temperature),
      'style': SupervisedContrastiveLoss(temperature),
      'genre': SupervisedContrastiveLoss(temperature),
    }

    callbacks = lambda: [
      tf.keras.callbacks.TerminateOnNaN(),
      tf.keras.callbacks.EarlyStopping(patience=40, verbose=1),
      tf.keras.callbacks.TensorBoard(logs, histogram_freq=1, write_graph=True),
      tf.keras.callbacks.ModelCheckpoint(weights, save_weights_only=True, save_best_only=True, verbose=1),
      ReduceLRBacktrack(min_lr=1e-6, best_path=weights, distributed_strategy=dst),
    ]

  class finetune:
    perform = os.environ.get('PERFORM_FT', 'true') == 'true'

    epochs = int(os.environ.get('EPOCHS_FT', '200'))
    layers = 0.4  # 40%
    freeze_bn = False

    train_steps = None
    valid_steps = None
    initial_epoch = int(os.environ.get('INITIAL_EPOCH_FT', '0'))
    
    learning_rate = float(os.environ.get('LR_FT', '0.0001'))
    
    optimizer_name = os.environ.get('OPT_FT', 'momentum')
    optimizer = get_optimizer(optimizer_name, learning_rate)

    logs = os.path.join(logs_dir, f'i3-supcon-baseline-opt:{optimizer_name}-lr:{learning_rate}-ft')
    weights = os.path.join(weights_dir, f'i3-supcon-baseline-opt:{optimizer_name}-lr:{learning_rate}-ft')
    exported = os.path.join(weights_dir, f'i3-supcon-baseline-opt:{optimizer_name}-lr:{learning_rate}-ex')

    callbacks = lambda: [
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
  
  backbone.trainable = False

  nn = supcon_encoder_multitask_network(
    input_tensor=SupconConfig.input_tensor,
    backbone=backbone,
    pooling=SupconConfig.model.pooling,
    artist_projection_head=ProjectionHead(name='artist_sadh', pr_ks=SupconConfig.model.pr_ks),
    style_projection_head=ProjectionHead(name='style_sadh', pr_ks=SupconConfig.model.pr_ks),
    genre_projection_head=ProjectionHead(name='genre_sadh', pr_ks=SupconConfig.model.pr_ks),
    name='painter-by-numbers/supcon'
  )

  nn.compile(
    optimizer=SupconConfig.training.optimizer,
    loss=SupconConfig.training.loss,
    loss_weights=SupconConfig.training.loss_weights
  )

  print('[SADN head fit] Network Summary')
  nn.summary()

### Training
# nn(tf.random.normal((DataConfig.batch_size, 299, 299, 3)), training=False);  # sanity check.
if SupconConfig.training.perform:
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
    callbacks=SupconConfig.training.callbacks(),
    override=SupconConfig.override,
    distributed_strategy=dst,
  )


### Fine-Tuning
if SupconConfig.finetune.perform:
  print('[SADN finetuning] Network Summary')
  nn.summary()

  if SupconConfig.training.perform:
    initial_epoch = len(report.history['loss'])
  else:
    initial_epoch = SupconConfig.finetune.initial_epoch

  with dst.scope():
    nn.load_weights(SupconConfig.training.weights)

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
    callbacks=SupconConfig.finetune.callbacks(),
    override=SupconConfig.override,
    distributed_strategy=dst,
  )

  print(f'Exporting model to {SupconConfig.finetune.exported}')
  nn.save(SupconConfig.finetune.exported, save_format='tf')

# endregion

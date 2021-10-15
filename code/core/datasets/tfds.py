from functools import partial
from typing import Any, Dict, List

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from .augment import augment_policy_fn, default_policy_fn


def load_dataset(
    dataset_name,
    data_dir,
):
  print(f'Loading dataset {dataset_name} into {data_dir}')

  (train, test), info = tfds.load(
    dataset_name,
    split=['train', 'test'],
    with_info=True,
    shuffle_files=True,
    data_dir=data_dir
  )
  return (train, test), info


def split_classes(
    dataset: tf.data.Dataset,
    classes_splits: List[List[int]],
):
  splits = []

  for c in classes_splits:
    splits.append(dataset.filter(lambda x: x in c))

  return splits


def prepare(
    dataset: tf.data.Dataset,
    batch_size: int,
    augment: bool = False,
    aug_config: Dict[str, Any] = None,
    buffer_size: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    preprocess_fn=None,
    randgen = None,
):
  if buffer_size == 'auto':
    buffer_size = tf.data.AUTOTUNE
  if num_parallel_calls == 'auto':
    num_parallel_calls = tf.data.AUTOTUNE
  
  policy = augment_policy_fn if augment else default_policy_fn
  policy = partial(policy, aug_config=aug_config, randgen=randgen, preprocess_fn=preprocess_fn)

  return (dataset
          .map(lambda x: (x['image'], x['label']), num_parallel_calls=num_parallel_calls)
          .batch(batch_size, drop_remainder=True)
          .map(policy, num_parallel_calls=tf.data.AUTOTUNE)
          .prefetch(buffer_size))

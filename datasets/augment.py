import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def default_policy_fn(images, labels, preprocess_fn, aug_config, randgen):
  return preprocess_fn(tf.cast(images, tf.float32)), labels


def augment_policy_fn(image, labels, preprocess_fn, aug_config, randgen):
  batch_size = tf.shape(image)[:1]

  seeds = randgen.make_seeds(7)

  angles = tf.random.stateless_uniform(batch_size, maxval=2*np.pi, seed=seeds[:, 6])
  image = tfa.image.rotate(image, angles=angles, fill_mode='reflect')

  image = tf.image.stateless_random_flip_left_right(image, seed=seeds[:, 0])
  image = tf.image.stateless_random_flip_up_down(image, seed=seeds[:, 1])
  image = tf.image.stateless_random_hue(image, aug_config['hue_delta'], seed=seeds[:, 2])
  image = tf.image.stateless_random_brightness(image, aug_config['brightness_delta'], seed=seeds[:, 3])
  image = tf.image.stateless_random_contrast(image, aug_config['contrast_lower'], aug_config['contrast_upper'], seed=seeds[:, 4])
  image = tf.image.stateless_random_saturation(image, aug_config['saturation_lower'], aug_config['saturation_upper'], seed=seeds[:, 5])

  return preprocess_fn(tf.cast(image, tf.float32)), labels

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D


@tf.keras.utils.register_keras_serializable("afhp")
class SupConProjectionHead(tf.keras.layers.Layer):
  """Supcon Projection Head

  Based on: https://github.com/google-research/google-research/blob/master/supcon/projection_head.py
  """

  def __init__(
      self,
      proj1_units=2048,
      proj2_units=128,
      normalize_output=True,
      pr_ks=None,
      **kwargs
  ):
    super().__init__(**kwargs)

    ki = tf.random_normal_initializer(stddev=.01)
    pr_ks = pr_ks or {}
    self.proj_1 = Dense(proj1_units, use_bias=True,  kernel_initializer=ki, activation='relu', **pr_ks)
    self.proj_2 = Dense(proj2_units, use_bias=False, kernel_initializer=ki, dtype="float32", **pr_ks)

    self.proj1_units = proj1_units
    self.proj2_units = proj2_units
    self.normalize_output = normalize_output

  def get_config(self):
    return {
      'proj1_units': self.proj1_units,
      'proj2_units': self.proj2_units,
      'normalize_output': self.normalize_output,
    }

  def call(self, x):
    y = self.proj_1(x)
    y = self.proj_2(y)

    if self.normalize_output:
      y = tf.math.l2_normalize(y, axis=1)

    return y


def supcon_encoder_network(
  input_tensor,
  backbone,
  pooling='avg',
  painter_projection_head=None,
  normalize_stem_output=True,
  head_proj_kwargs=None,
  **kwargs
):
  y = backbone(input_tensor)
  if pooling == 'avg':
    y = GlobalAveragePooling2D(name='avg_pool')(y)
  elif pooling == "max":
    y = GlobalMaxPooling2D(name='max_pool')(y)
  else:
    raise ValueError(f"Unknown global pooling player {pooling}. Options are {('avg', 'max')} (default='avg').")

  if normalize_stem_output:
    y = tf.math.l2_normalize(y, axis=1)

  painter_projection_head = painter_projection_head or SupConProjectionHead(name='painter_project_head', pr_ks=head_proj_kwargs)
  painter = painter_projection_head(y)

  return tf.keras.Model(inputs=input_tensor, outputs=painter, **kwargs)


def supcon_encoder_multitask_network(
  input_tensor,
  backbone,
  pooling='avg',
  painter_projection_head=None,
  style_projection_head=None,
  genre_projection_head=None,
  normalize_stem_output=True,
  **kwargs
):
  painter_projection_head = painter_projection_head or SupConProjectionHead(name='painter_project_head')
  style_projection_head = style_projection_head or SupConProjectionHead(name='style_project_head')
  genre_projection_head = genre_projection_head or SupConProjectionHead(name='genre_project_head')

  y = backbone(input_tensor)

  if pooling == 'avg':
    y = GlobalAveragePooling2D(name='avg_pool')(y)
  elif pooling == "max":
    y = GlobalMaxPooling2D(name='max_pool')(y)
  else:
    raise ValueError(f"Unknown global pooling player {pooling}. Options are {('avg', 'max')} (default='avg').")

  if normalize_stem_output:
    y = tf.math.l2_normalize(y, axis=1)

  painter = painter_projection_head(y)
  style = style_projection_head(y)
  genre = genre_projection_head(y)

  return tf.keras.Model(
    inputs=input_tensor,
    outputs={
      'painter': painter,
      'style': style,
      'genre': genre
    },
    **kwargs)


@tf.keras.utils.register_keras_serializable("afhp")
class SADH(tf.keras.layers.Layer):
  def __init__(
      self,
      discr_units=1,
      proj1_units=2048,
      proj2_units=2048,
      **kwargs
  ):
    super().__init__(**kwargs)

    self.proj1_units = proj1_units
    self.proj2_units = proj2_units
    self.discr_units = discr_units

    self.proj_1 = Dense(proj1_units, activation='relu', name='am/proj_1')
    self.proj_2 = Dense(proj2_units, activation='relu', name='am/proj_1')
    self.discr  = Dense(discr_units, name='am/predictions')

  def get_config(self):
    return {
      'discr_units': self.discr_units,
      'proj1_units': self.proj1_units,
      'proj2_units': self.proj2_units,
    }

  def call(self, inputs):
    y = tf.keras.layers.multiply([self.proj_2(self.proj_1(x)) for x in inputs], name='joint_multiply')
    y = self.discr(y)

    return y


@tf.keras.utils.register_keras_serializable("afhp")
class SupervisedContrastiveLoss(tf.losses.Loss):
  def __init__(self, temperature=1.0, **kwargs):
    super().__init__(**kwargs)
    self.temperature = temperature

  def call(self, labels, features):
    logits = tf.matmul(features, features, transpose_b=True)
    loss = tfa.losses.npairs_loss(
        tf.squeeze(labels),
        logits / self.temperature
    )

    print(f'[SupervisedContrastiveLoss tracing] labels:{labels.shape} features:{features.shape} loss:{loss.shape}')
    return loss

  def get_config(self):
    return {'temperature': self.temperature}

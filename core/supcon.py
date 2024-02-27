import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D


@tf.keras.utils.register_keras_serializable("afhp")
def supcon_projection_head(
  inputs: tf.Tensor,
  name: str,
  proj1_units: int = 2048,
  proj2_units: int = 128,
  **kwargs,
):
  """Supcon Projection Head

  Based on: https://github.com/google-research/google-research/blob/master/supcon/projection_head.py
  """
  kwargs = kwargs or {}
  kwargs["kernel_initializer"] = tf.random_normal_initializer(stddev=.01)

  p1 = Dense(proj1_units, name=f"{name}_proj1", use_bias=True, activation='relu', **kwargs)(inputs)
  p2 = Dense(proj2_units, name=f"{name}_proj2", use_bias=False, dtype="float32", **kwargs)(p1)

  return p2


def supcon_encoder(
  input_tensor,
  backbone,
  pooling='avg',
  normalize_stem_output=True,
  proj_kwargs=None,
  **kwargs,
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

  proj_kwargs = proj_kwargs or {}
  y = supcon_projection_head(y, name='painter', **proj_kwargs)

  return tf.keras.Model(inputs=input_tensor, outputs=y, **kwargs)


def supcon_encoder_mh(
  input_tensor,
  backbone,
  pooling='avg',
  normalize_stem_output=True,
  proj_kwargs=None,
  **kwargs,
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

  proj_kwargs = proj_kwargs or {}
  painter = supcon_projection_head(y, name='painter', **proj_kwargs)
  style = supcon_projection_head(y, name='style', **proj_kwargs)
  genre = supcon_projection_head(y, name='genre', **proj_kwargs)

  return tf.keras.Model(
    inputs=input_tensor,
    outputs={
      'painter': painter,
      'style': style,
      'genre': genre
    },
    **kwargs)


@tf.keras.utils.register_keras_serializable("afhp")
class SupervisedContrastiveLoss(tf.losses.Loss):
  def __init__(self, temperature=1.0, l2_norm: bool = True, **kwargs):
    super().__init__(**kwargs)
    self.temperature = temperature
    self.l2_norm = l2_norm

  def call(self, labels, features):
    if self.l2_norm:
      features = tf.math.l2_normalize(features, axis=1)

    logits = tf.matmul(features, features, transpose_b=True)
    loss = tfa.losses.npairs_loss(
        tf.squeeze(labels),
        logits / self.temperature
    )

    print(f'[SupervisedContrastiveLoss tracing] labels:{labels.shape} features:{features.shape} loss:{loss.shape}')
    return loss

  def get_config(self):
    return {'temperature': self.temperature, "l2_norm": self.l2_norm}

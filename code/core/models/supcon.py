import tensorflow as tf
from tensorflow.keras.layers import Dense


class ProjectionHead(tf.keras.layers.Layer):
  """Supcon Projection Head

  Based on: https://github.com/google-research/google-research/blob/master/supcon/projection_head.py
  """

  def __init__(
      self,
      proj1_units=2048,
      proj2_units=128,
      normalize_output=True,
      proj_kwargs=None,
      **kwargs
  ):
    super().__init__(**kwargs)

    proj_kwargs = proj_kwargs or {}

    ki = tf.random_normal_initializer(stddev=.01)
    self.proj_1 = Dense(proj1_units, activation='relu', kernel_initializer=ki, **proj_kwargs)
    self.proj_2 = Dense(proj2_units, activation='relu', kernel_initializer=ki, **proj_kwargs)

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

    y = tf.math.l2_normalize(y, axis=1)

    return y


def supcon_encoder_network(
  input_tensor,
  backbone,
  artist_projection_head=None,
  normalize_stem_output=True,
  head_proj_kwargs=None,
  **kwargs
):
  y = backbone(input_tensor)

  if normalize_stem_output:
    y = tf.math.l2_normalize(y, axis=1)
  
  artist_projection_head = artist_projection_head or ProjectionHead(name='artist_project_head', proj_kwargs=head_proj_kwargs)
  artist = artist_projection_head(y)

  return tf.keras.Model(inputs=input_tensor, outputs=artist, **kwargs)


def supcon_encoder_multitask_network(
  input_tensor,
  backbone,
  artist_projection_head=None,
  style_projection_head=None,
  genre_projection_head=None,
  normalize_stem_output=True,
  **kwargs
):
  artist_projection_head = artist_projection_head or ProjectionHead(name='artist_project_head')
  style_projection_head = style_projection_head or ProjectionHead(name='style_project_head')
  genre_projection_head = genre_projection_head or ProjectionHead(name='genre_project_head')

  y = backbone(input_tensor)

  if normalize_stem_output:
    y = tf.math.l2_normalize(y, axis=1)
  
  artist = artist_projection_head(y)
  style = style_projection_head(y)
  genre = genre_projection_head(y)

  return tf.keras.Model(inputs=input_tensor, outputs=[artist, style, genre], **kwargs)


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

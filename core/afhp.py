import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Concatenate, Dense, LeakyReLU


def build_generator(
    noise: tf.keras.Input,
    context: tf.keras.Input,
    features: int,
) -> Model:
  x = Concatenate(name='concat1')([noise, context])
  y = Dense(1024, name='fc1')(x)
  y = LeakyReLU(0.2, name='a1')(y)
  y = Dense(features, name='fc2')(y)
  y = Activation('relu', name='a2')(y)

  return Model([noise, context], y, name='wgan_features_d')


def build_discriminator(
    features: tf.keras.Input,
    context: tf.keras.Input,
) -> Model:
  x = Concatenate(name='concat1')([features, context])
  y = Dense(1024, name='fc1')(x)
  y = LeakyReLU(0.2, name='a1')(y)
  y = Dense(1, name='predictions')(y)

  return Model([features, context], y, name='wgan_features_g')


def build(noise: int, features: int, **kwargs):
  noise_tensor = tf.keras.Input([noise], name='noise')
  features_tensor = tf.keras.Input([features], name='features')
  context_tensor = tf.keras.Input([features], name='context')

  G = build_generator(
    noise_tensor,
    context_tensor,
    features=features
  )

  D = build_discriminator(
    features_tensor,
    context_tensor
  )

  return AFHP(
    discriminator=D,
    generator=G,
    noise=noise,
    **kwargs,
  )


@tf.keras.utils.register_keras_serializable("afhp")
class AFHP(Model):
  def __init__(
      self,
      discriminator: Model,
      generator: Model,
      k: int = 10,
      noise: int = 512,
      d_steps: int = 3,
      gp_weight: float = 10.0,
      ac_weight: float = 1.0,
      cf_weight: float = 1.0,
      **kwargs
  ):
    super(AFHP, self).__init__(**kwargs)
    self.D = discriminator
    self.G = generator
    self.d_steps = d_steps
    self.gp_weight = gp_weight
    self.ac_weight = ac_weight
    self.cf_weight = cf_weight
    self.k = k
    self.noise = noise

  def compile(self, d_opt, g_opt, metric_trackers=None, **kwargs):
    super(AFHP, self).compile(**kwargs)
    self.d_opt = d_opt
    self.g_opt = g_opt

    self.metric_trackers = metric_trackers or {}

  @property
  def metrics(self):
    return list(self.metric_trackers.values())

  def update_metric_trackers(self, **metrics):
    for m, v in metrics.items():
      if m not in self.metric_trackers:
        self.metric_trackers[m] = tf.keras.metrics.Mean(name=m)
      self.metric_trackers[m].update_state(v)

  def build_context(self, x):
    context = tf.reduce_mean(x, axis=1)
    context = tf.repeat(context, self.k, axis=0)

    x = tf.reshape(x, (-1, x.shape[-1]))

    return x, context

  def call(self, inputs, training=None, mask=None):
    """Defer to Generator, hallucinating features in a given context given real samples.
    """
    x, context = self.build_context(inputs)

    batch_size = tf.shape(x)[0]

    z1 = tf.random.normal(shape=(batch_size, self.noise))

    x_fake = self.G([z1, context], training=training)
    p_fake = self.D([x_fake, context], training=training)

    return x_fake, p_fake

  def train_step(self, inputs):
    x_real, y_real, x_query, y_query = inputs
    x_real, context = self.build_context(x_real)

    batch_size = tf.shape(x_real)[0]

    self.d_train_step(x_real, context, batch_size)

    iterations = self.d_opt.iterations

    if iterations % self.d_steps == 0:
      self.g_train_step(x_real, y_real, x_query, y_query, context, batch_size)

    return self.get_metrics_result()

  # @tf.function(reduce_retracing=True)
  def d_train_step(self, x_real, context, batch_size):
    z1 = tf.random.normal(shape=(batch_size, self.noise))

    with tf.GradientTape() as tape:
      x_fake = self.G([z1, context], training=True)
      p_fake = self.D([x_fake, context], training=True)
      p_real = self.D([x_real, context], training=True)

      gp_loss = gradient_penalty(x_real, x_fake, context, batch_size, self.D)
      d_loss = (discriminator_loss(p_real, p_fake) + self.gp_weight * gp_loss)

    g = tape.gradient(d_loss, self.D.trainable_variables)
    update_op = self.d_opt.apply_gradients(zip(g, self.D.trainable_variables))

    self.update_metric_trackers(
      gp_loss=gp_loss,
      d_loss=d_loss
    )

    return update_op

  # @tf.function(reduce_retracing=True)
  def g_train_step(self, x_real, y_real, x_query, y_query, context, batch_size):
    z1 = tf.random.normal(shape=(batch_size, self.noise))
    z2 = tf.random.normal(shape=(batch_size, self.noise))

    with tf.GradientTape() as tape:
      x_fake1 = self.G([z1, context], training=True)
      x_fake2 = self.G([z2, context], training=True)

      p_fake1 = self.D([x_fake1, context], training=True)
      p_fake2 = self.D([x_fake2, context], training=True)

      g_loss = generator_loss(p_fake1, p_fake2)
      a_loss = 1 / anti_collapse_regularizer(z1, z2, x_fake1, x_fake2)
      c_loss = classifier_loss(x_fake1[::self.k], x_fake2[::self.k], y_real, x_query, y_query)
      t_loss = (
        g_loss
        + self.ac_weight * a_loss
        + self.cf_weight * c_loss
      )

    g = tape.gradient(t_loss, self.G.trainable_variables)
    update_op = self.g_opt.apply_gradients(zip(g, self.G.trainable_variables))

    self.update_metric_trackers(
      g_loss=g_loss,
      a_loss=a_loss,
      t_loss=t_loss,
      c_loss=c_loss
    )

    return update_op

  def test_step(self, inputs):
    x_real, y_real, x_query, y_query = inputs
    x_real, context = self.build_context(x_real)

    batch_size = tf.shape(x_real)[0]

    z = tf.random.normal(shape=(2 * batch_size, self.noise))
    z1, z2 = tf.split(z, 2)

    x_fake = self.G([z, tf.tile(context, (2, 1))], training=False)
    x_fake1, x_fake2 = tf.split(x_fake, 2)

    x = tf.concat((x_real, x_fake), axis=0)

    p = self.D([x, tf.tile(context, (3, 1))], training=False)
    p_real, p_fake1, p_fake2 = tf.split(p, 3)

    d_loss = discriminator_loss(p_real, p_fake1)
    gp_loss = gradient_penalty(x_real, x_fake1, context, batch_size, self.D)

    g_loss = generator_loss(p_fake1, p_fake2)
    a_loss = 1 / anti_collapse_regularizer(z1, z2, x_fake1, x_fake2)
    c_loss = classifier_loss(x_fake1[::self.k], x_fake2[::self.k], y_real[::self.k], x_query, y_query)
    t_loss = (
      g_loss
      + self.ac_weight * a_loss
      + c_loss
    )

    self.update_metric_trackers(
      g_loss=g_loss,
      a_loss=a_loss,
      c_loss=c_loss,
      t_loss=t_loss,
      d_loss=d_loss,
      gp_loss=gp_loss,
    )

    return self.get_metrics_result()


def discriminator_loss(p_real, p_fake):
  return (tf.reduce_mean(p_fake)
          - tf.reduce_mean(p_real))


def generator_loss(p1_fake, p2_fake):
  return -.5*(tf.reduce_mean(p1_fake)
              + tf.reduce_mean(p2_fake))


def gradient_penalty(real_images, fake_images, context, batch_size, D):
  a = tf.random.uniform([batch_size, 1])
  x = a * real_images + (1-a) * fake_images

  with tf.GradientTape() as tape:
      tape.watch(x)
      p = D([x, context], training=False)

  g = tape.gradient(p, x)
  g = tf.norm(g, axis=1)
  g = tf.reduce_mean((g - 1.0) ** 2)

  return g

# Eq (7)

def cosine_proximity(y_true, y_pred, axis=-1):
  y_true = tf.linalg.l2_normalize(y_true, axis=axis)
  y_pred = tf.linalg.l2_normalize(y_pred, axis=axis)
  return tf.reduce_sum(y_true * y_pred, axis=axis)

def anti_collapse_regularizer(z1, z2, x1_fake, x2_fake):
  return tf.reduce_mean(
    (1 - cosine_proximity(x1_fake, x2_fake))
    / (1 - cosine_proximity(z1, z2))
  )

# Eq (5)

def classifier_loss(x1_fake, x2_fake, y_real, x_q, y_q):
  cos = cosine_proximity(x1_fake, x_q[:, tf.newaxis, ...])
  loss_z1 = tf.nn.sparse_softmax_cross_entropy_with_logits(y_q, cos)

  cos = cosine_proximity(x2_fake, x_q[:, tf.newaxis, ...])
  loss_z2 = tf.nn.sparse_softmax_cross_entropy_with_logits(y_q, cos)

  return tf.reduce_mean(loss_z1) + tf.reduce_mean(loss_z2)

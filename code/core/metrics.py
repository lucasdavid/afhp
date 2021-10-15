import tensorflow as tf
import tensorflow_addons as tfa


class FromLogitsMixin:
  def __init__(self, from_logits=False, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.from_logits = from_logits

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.from_logits:
      y_pred = tf.nn.sigmoid(y_pred)
    return super().update_state(y_true, y_pred, sample_weight)


class BinaryAccuracy(FromLogitsMixin, tf.metrics.BinaryAccuracy):
  ...

class TruePositives(FromLogitsMixin, tf.metrics.TruePositives):
  ...

class FalsePositives(FromLogitsMixin, tf.metrics.FalsePositives):
  ...

class TrueNegatives(FromLogitsMixin, tf.metrics.TrueNegatives):
  ...

class FalseNegatives(FromLogitsMixin, tf.metrics.FalseNegatives):
  ...

class Precision(FromLogitsMixin, tf.metrics.Precision):
  ...

class Recall(FromLogitsMixin, tf.metrics.Recall):
  ...

class F1Score(FromLogitsMixin, tfa.metrics.F1Score):
  ...


class SupervisedContrastiveLoss(tf.losses.Loss):
  def __init__(self, temperature=1.0, **kwargs):
    super().__init__(**kwargs)
    self.temperature = temperature

  def call(self, labels, features):
    print(f'[SupervisedContrastiveLoss tracing] labels:{labels.shape} features:{features.shape}')

    logits = tf.matmul(features, features, transpose_b=True)

    return tfa.losses.npairs_loss(
        tf.squeeze(labels),
        logits / self.temperature
    )

  def get_config(self):
    return {'temperature': self.temperature}

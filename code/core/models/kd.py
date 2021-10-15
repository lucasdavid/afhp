import tensorflow as tf

from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

from ..metrics import *


def build_teacher(
    backbone,
    input_tensor,
    classes,
    dropout_rate,
    fine_tune_layers,
    freeze_batch_norm,
    weights = None,
    trainable = False,
    name = None
):
  teacher_backbone = backbone(
    input_tensor=input_tensor,
    include_top=False
  )

  teacher_backbone.trainable = False

  teacher = build_classifier(
    teacher_backbone,
    input_tensor,
    classes,
    dropout_rate=dropout_rate,
    name = name or 'teacher')

  unfreeze_top_layers(
    teacher_backbone,
    fine_tune_layers,
    freeze_batch_norm
  )

  if weights:
    teacher.load_weights(weights)

  teacher.trainable = trainable

  return teacher


def build_classifier(
    backbone,
    input_tensor,
    classes,
    dropout_rate=None,
    name=None,
):
  y = backbone(input_tensor)
  y = GlobalAveragePooling2D(name='avg_pool')(y)
  y = Dropout(rate=dropout_rate, name='top_dropout')(y)
  y = Dense(classes, name='predictions')(y)

  return tf.keras.Model(
    inputs=input_tensor,
    outputs=y,
    name=name
  )


def unfreeze_top_layers(
    model: tf.keras.Model,
    layers: float,
    freeze_bn: bool
):
  if not layers:
    model.trainable = False
    return

  model.trainable = True

  frozen_layer_ix = int((1-layers) * len(model.layers))
  for ix, l in enumerate(model.layers):
    l.trainable = (ix > frozen_layer_ix and
                   (not isinstance(l, tf.keras.layers.BatchNormalization) or
                    not freeze_bn))
  
  print(f'Unfreezing {layers:.0%} layers. Bottom-most is the {frozen_layer_ix}-nth layer ({model.layers[frozen_layer_ix].name}).')


class FitNet(tf.keras.Model):
  def __init__(
      self,
      student,
      teacher,
      alpha=0.1,
      beta=1.0,
      temperature=3,
      global_batch_size=None,
      **kwargs
  ):
    super().__init__(**kwargs)
    self.student = student
    self.teacher = teacher

    self.alpha = alpha
    self.beta = beta
    self.temperature = temperature
    self.global_batch_size = global_batch_size
  
  def get_config(self):
    config = super().get_config()
    config.update(dict(
      alpha=self.alpha,
      beta=self.beta,
      temperature=self.temperature,
      global_batch_size=self.global_batch_size))

    return config
  
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              distillation_loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None,
              **kwargs):
    super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    self.distillation_loss = distillation_loss or tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
  
  def call(self, inputs):
    return self.student(inputs)[0]

  def train_step(self, data):
    if len(data) == 3:
      x, y, sample_weight = data
    else:
      sample_weight = None
      x, y = data
    
    tp, *t_signals = self.teacher(x, training=False)

    with tf.GradientTape() as tape:
      sp, *s_signals = self.student(x, training=True)

      features_loss = tf.add_n([hint_loss(tf.stop_gradient(t), s) for t, s in zip(t_signals, s_signals)])

      distillation_loss = self.distillation_loss(
          tf.nn.softmax(tp / self.temperature, axis=1),
          tf.nn.softmax(sp / self.temperature, axis=1),
      )

      student_loss = self.compiled_loss(y, sp, sample_weight=sample_weight)

      loss = (self.alpha * student_loss
              + (1 - self.alpha) * distillation_loss
              + self.beta * features_loss)
      loss = tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)

    self.optimizer.minimize(loss, self.student.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(y, sp)

    results = {m.name: m.result() for m in self.metrics}
    results.update({
      "student_loss": tf.reduce_mean(student_loss),
      "distillation_loss": tf.reduce_mean(distillation_loss),
      'features_loss': tf.reduce_mean(features_loss)
    })

    return results

  def test_step(self, data):
    x, y = data
    sp, *s_signals = self.student(x, training=False)
    student_loss = self.compiled_loss(y, sp, regularization_losses=self.losses)
    self.compiled_metrics.update_state(y, sp)

    results = {m.name: m.result() for m in self.metrics}
    results.update({"student_loss": tf.reduce_mean(student_loss)})

    return results


def build_fitnet(
    teacher,
    student,
    hints,
    backbone_layer_index: int = 1,
    global_batch_size: int = None,
    name: str = None
):
  teacher_signals = [teacher.layers[backbone_layer_index].get_layer(h['teacher']).output for h in hints]
  student_signals = [student.layers[backbone_layer_index].get_layer(h['student']).output for h in hints]

  teacher_fb = tf.keras.Model(
    teacher.inputs,
    teacher.outputs + teacher_signals,
    name='teacher_fb'
  )

  student_fb = tf.keras.Model(
    student.inputs,
    student.outputs + student_signals,
    name='student_fb'
  )

  fn = FitNet(
    student_fb,
    teacher_fb,
    alpha=0.1,
    beta=1.0,
    temperature=10,
    global_batch_size=global_batch_size,
    name=name
  )

  return fn

import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE

from core.utils import get_extractor_params, get_afhp_params
from datasets import pbn


def run(
    train_info: "pd.DataFrame",
    all_info: "pd.DataFrame",
    distributed_strategy,
    args,
):
  print("[AFHP Training]")
  from core import afhp, fsl
  
  N = 5  # N-Way (classes)
  K = 10 # K-Shot (samples p/ class)
  Q = 10 # Query (samples p/ query)
  R = 3  # Repetitions

  _, _, _, _, _, _, _, _, _, ENC_NAME, _, _ = get_extractor_params(args)
  FEATURES_LAYER = args.backbone_features_layer
  EXTRACTOR_NAME = f"{ENC_NAME}_{FEATURES_LAYER}"

  PROBLEM, EPOCHS, LR, D_STEPS, NOISE, FEATURES, TASK_MODE, NAME, WEIGHTS, WEIGHTS_EXIST = get_afhp_params(args)

  fnames, features = pbn.load_features_file(args.preds_dir, EXTRACTOR_NAME, patches=args.patches_infer, parts=args.features_parts)
  is_train = np.isin(fnames, train_info.filename)

  fnames_train, feats_train = fnames[is_train], features[is_train]
  fnames_test, feats_test   = fnames[~is_train], features[~is_train]
  csamples_train, CLASSES_TRAIN = fsl.groupby_class(fnames_train, feats_train, all_info, class_col=PROBLEM)
  csamples_test, CLASSES_TEST = fsl.groupby_class(fnames_test, feats_test, all_info, class_col=PROBLEM)
  train_ds = fsl.build_support_query_tf_dataset(csamples_train, K, N, Q, FEATURES, TASK_MODE)
  test_ds = fsl.build_support_query_tf_dataset(csamples_test, K, N, Q, FEATURES, TASK_MODE)
  train_steps = R * len(CLASSES_TRAIN) // N  # ~3 * 1000 / 5 = 600
  test_steps = R * len(CLASSES_TEST) // N    # ~3 * 150 / 5 = 90

  with distributed_strategy.scope():
    afhp_model = afhp.build(
      noise=NOISE,
      features=FEATURES,
      d_steps=D_STEPS,
      gp_weight=10.0,
      ac_weight=1.0,
      cf_weight=1.0,
      name=f"afhp_{ENC_NAME}",
    )

    learning_rate_D = tf.keras.optimizers.schedules.ExponentialDecay(
      LR,
      decay_steps=EPOCHS * train_steps,
      decay_rate=0.5,
      staircase=True)

    learning_rate_G = tf.keras.optimizers.schedules.ExponentialDecay(
      LR,
      decay_steps=EPOCHS * np.ceil(train_steps // D_STEPS),
      decay_rate=0.5,
      staircase=True)
    
    afhp_model.compile(
      d_opt = tf.optimizers.Adam(learning_rate=learning_rate_D, beta_1=0.9, beta_2=0.999),
      g_opt = tf.optimizers.Adam(learning_rate=learning_rate_G, beta_1=0.9, beta_2=0.999),
      run_eagerly=False,
      jit_compile=True,
    )

  if not args.step_afhp_train or WEIGHTS_EXIST and not args.override:
    _ = afhp_model.predict(np.random.randn(1, K, FEATURES), verbose=0)

    with distributed_strategy.scope():
      print(f"loading weights from {WEIGHTS}")
      afhp_model.load_weights(WEIGHTS)

    history = None
  else:
    try:
      print("=" * 65)
      print("Training AFHP model")
      os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)

      history = afhp_model.fit(
        train_ds,
        epochs=args.afhp_epochs,
        steps_per_epoch=train_steps,
        validation_data=test_ds,
        validation_steps=test_steps,
        callbacks=_callbacks(afhp_model, args),
        verbose=1,
      )
    except KeyboardInterrupt:
      print('\n  interrupted')
      history = afhp_model.history
    else:
      print('\n  done')

    if args.afhp_persist:
      afhp_model.save_weights(WEIGHTS)

  if args.step_afhp_test:
    save_path =  os.path.join(args.logs_dir, afhp_model.name, "examples-embeddings.png")
    plot_embeddings(afhp_model.G, train_ds, test_ds, save_path, NOISE)


def _callbacks(model, args):
  return [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.TensorBoard(os.path.join(args.logs_dir, model.name), histogram_freq=1, write_graph=True, profile_batch=(10, 20)),
  ]


#region Visualizing Hallucinated Features


def plot_embeddings(G, train_ds, test_ds, save_path, noise_features, extra_k=50):
  import seaborn as sns
  import matplotlib.pyplot as plt
  sns.set(style="whitegrid")

  fig = plt.figure(figsize=(16, 9))
  plt.subplot(121)
  x, y, is_real = _gen_fake_features(G, train_ds, noise_features, extra_k=extra_k)
  _vis_fake_features(x, y, is_real)
  plt.xlabel('Training')

  plt.subplot(122)
  x, y, is_real = _gen_fake_features(G, test_ds, noise_features, extra_k=extra_k)
  _vis_fake_features(x, y, is_real)
  plt.xlabel('Testing')

  plt.tight_layout()
  fig.savefig(save_path)
  fig.clf()
  del fig


def _gen_fake_features(G, dataset, noise_features, extra_k=50):
  samples = []

  (x_real, y_real, x_query, y_query), = list(dataset.take(1))

  # Hallucinate GC new samples using the generator.
  context = tf.reduce_mean(x_real, axis=1) # [5,10,2048] -> [5,2048]
  context = tf.repeat(context, extra_k, axis=0)

  z1 = tf.random.normal((context.shape[0], noise_features))
  x1_fake = G.predict_on_batch([z1, context])
  x1_fake = tf.convert_to_tensor(x1_fake)

  y_fake = tf.repeat(y_real, extra_k, axis=0)

  B, K, F = x_real.shape

  x_real = tf.reshape(x_real, (-1, F))
  y_real = tf.repeat(y_real, K, axis=0)

  samples.append((x_real, y_real, tf.ones_like(y_real)))
  samples.append(((x1_fake, y_fake, tf.zeros_like(y_fake))))

  x, y, is_real = (tf.concat(e, axis=0).numpy() for e in zip(*samples))

  return x, y, is_real.astype(bool)


def _vis_fake_features(x, y, is_real):
  import seaborn as sns
  import matplotlib.pyplot as plt

  with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    z = TSNE(init='pca', learning_rate='auto').fit_transform(x)
    y = np.char.add("class ", y.astype("str"))

    sns.scatterplot(x=z[is_real, 0], y=z[is_real, 1], hue=y[is_real], marker='s', label='Original', alpha=0.8, legend=False)
    sns.scatterplot(x=z[~is_real, 0], y=z[~is_real, 1], hue=y[~is_real], label='Generated', alpha=0.8, legend='brief')
    plt.legend()

# endregion

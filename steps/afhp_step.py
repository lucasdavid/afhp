import os

import numpy as np
import tensorflow as tf
from keras import callbacks as callbacks_mod

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

  ENC_NAME = get_extractor_params(args)[-3]
  FEATURES_LAYER = args.backbone_features_layer
  EXTRACTOR_NAME = f"{ENC_NAME}_{FEATURES_LAYER}"

  PROBLEM, EPOCHS, LR, D_STEPS, NOISE, FEATURES, TASK_MODE, NAME, WEIGHTS, WEIGHTS_EXIST = get_afhp_params(args)

  fnames, features = pbn.load_features_file(args.preds_dir, EXTRACTOR_NAME, patches=args.patches_test, parts=args.features_parts)
  is_train = np.isin(fnames, train_info.filename)

  fnames_train, feats_train = fnames[is_train], features[is_train]
  fnames_test, feats_test   = fnames[~is_train], features[~is_train]

  _, n_pts, n_feats = feats_train.shape

  fmin, fmax = feats_train.min(), feats_train.max()

  feats_train -= fmin
  feats_train /= fmax
  feats_test -= fmin
  feats_test /= fmax

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
      gp_w=args.afhp_gp_w,
      ac_w=args.afhp_ac_w,
      cf_w=args.afhp_cf_w,
      name=NAME,
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
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        callbacks=_callbacks(afhp_model, args),
        use_multiprocessing=False,
        verbose=1,
      )
    except KeyboardInterrupt:
      print('\n  interrupted')
      history = afhp_model.history
    else:
      print('\n  done')

    if args.afhp_persist:
      afhp_model.save_weights(WEIGHTS)
      print(f"AFHP model saved at {WEIGHTS}")

  save_path =  os.path.join(args.logs_dir, afhp_model.name, "examples-embeddings.png")
  afhp.plot_embeddings(afhp_model.G, train_ds, test_ds, save_path, NOISE)
  print(f"Embedding representation saved at {save_path}")

  return afhp_model, (csamples_train, CLASSES_TRAIN), (csamples_test, CLASSES_TEST)


def _callbacks(model, args):
  return [
    callbacks_mod.TerminateOnNaN(),
    callbacks_mod.TensorBoard(os.path.join(args.logs_dir, model.name)),
    callbacks_mod.BackupAndRestore(os.path.join(args.logs_dir, model.name, "backup")),
  ]

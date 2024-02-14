from math import ceil
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from core.utils import unfreeze_top_layers, get_extractor_params

import datasets.pbn


def build_optimizer(name, lr):
  if name == "adam":
    return tf.keras.optimizers.Adam(learning_rate=lr)
  if name == "sgd":
    return tf.keras.optimizers.SGD(learning_rate=lr)
  if name == "momentum":
    return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)


def build_vanilla(name, size, classes, backbone_name):
  backbone_fn = getattr(tf.keras.applications, backbone_name)
  backbone = backbone_fn(include_top=False, weights="imagenet", input_shape=[size, size, 3], pooling=None)
  model = tf.keras.Sequential([
    backbone,
    tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
    tf.keras.layers.Dense(len(classes), activation="softmax", name="predictions", dtype="float32"),
  ], name=name)

  return model, backbone


def vanilla(
    info: "pd.DataFrame",
    all_info: "pd.DataFrame",
    distributed_strategy,
    args,
):
  SEED, B_TRAIN, S, P, W, BACKBONE, EPOCHS_HE, EPOCHS_FT, TARGET, NAME, WEIGHTS, WEIGHTS_EXIST = get_extractor_params(args)

  train_info, valid_info = train_test_split(info, test_size=args.backbone_valid_split, stratify=info[TARGET], random_state=SEED)

  train_ds = datasets.pbn.PbNPatchesSequence(train_info, train_info[TARGET], B_TRAIN, patch_size=S, patches=P, augment=True)
  valid_ds = datasets.pbn.PbNPatchesSequence(valid_info, valid_info[TARGET], B_TRAIN, patch_size=S, patches=P, target_encoder=train_ds.target_encoder)
  C = train_ds.painters
  print(f"painters={len(C)} train={len(train_info)} train_steps={len(train_ds)} valid={len(valid_info)} train_steps={len(valid_ds)}")

  with distributed_strategy.scope():
    model, backbone = build_vanilla(NAME, S, C, BACKBONE)
    backbone.trainable = False

    model.compile(
      optimizer=build_optimizer(args.backbone_optimizer, args.backbone_train_lr),
      loss=tf.losses.SparseCategoricalCrossentropy(),
      metrics=_metrics(),
      jit_compile=args.jit_compile,
    )
    model.summary()

  if EPOCHS_HE > 0 and (args.override or not WEIGHTS_EXIST):
    print("=" * 65)
    print("Training Head")
    model.fit(train_ds, epochs=EPOCHS_HE, callbacks=_callbacks(model, "head", args), workers=W, verbose=1)

  if EPOCHS_FT > 0:
    print("=" * 65)
    print("Fine-tuning entire model")
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)

    with distributed_strategy.scope():
      unfreeze_top_layers(
        backbone,
        args.backbone_finetune_layers,
        args.backbone_freezebn,
      )

      learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.backbone_finetune_lr,
        decay_steps=len(train_ds) * EPOCHS_FT,
      )
      model.compile(
        optimizer=build_optimizer(args.backbone_optimizer, learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=_metrics(),
        jit_compile=args.jit_compile,
      )

    if WEIGHTS_EXIST and not args.override:
      print(f"  Previous training found. Skipping training and loading weights from `{WEIGHTS}`.")
    else:
      callbacks = _callbacks(model, "finetune", args) + [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(WEIGHTS, save_weights_only=True, save_best_only=True, verbose=1),
      ]
      model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS_FT, callbacks=callbacks, workers=W, verbose=1)

  if args.step_features_infer:
    inference(model, all_info, distributed_strategy, args, NAME, WEIGHTS, TARGET, W)


def supcon(
    info: "pd.DataFrame",
    all_info: "pd.DataFrame",
    distributed_strategy,
    args,
):
  from core import supcon

  SEED, B_TRAIN, S, P, W, BACKBONE, EPOCHS_HE, EPOCHS_FT, TARGET, NAME, WEIGHTS, WEIGHTS_EXIST = get_extractor_params(args)

  if EPOCHS_HE > 0:
    print(f"SupCon requires training from all layers. Setting backbone_train_epochs = 0 (previous = {EPOCHS_HE}).")
    EPOCHS_HE = 0

  train_info, valid_info = train_test_split(info, test_size=args.backbone_valid_split, stratify=info[TARGET], random_state=SEED)

  train_ds = datasets.pbn.PbNPatchesSequence(train_info, train_info[TARGET], B_TRAIN, patch_size=S, patches=P, augment=True)
  valid_ds = datasets.pbn.PbNPatchesSequence(valid_info, valid_info[TARGET], B_TRAIN, patch_size=S, patches=P, target_encoder=train_ds.target_encoder)

  C = train_ds.painters
  print(f"painters={len(C)} train={len(train_info)} train_steps={len(train_ds)} valid={len(valid_info)} train_steps={len(valid_ds)}")

  with distributed_strategy.scope():
    backbone_fn = getattr(tf.keras.applications, BACKBONE)
    backbone = backbone_fn(include_top=False, weights="imagenet", input_shape=[S, S, 3], pooling=None)
    model = supcon.supcon_encoder_network(backbone.input, backbone, name=NAME)

  if EPOCHS_FT > 0:
    print("=" * 65)
    print("Fine-tuning entire model")
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)

    with distributed_strategy.scope():
      unfreeze_top_layers(
        backbone,
        args.backbone_finetune_layers,
        args.backbone_freezebn,
      )

      # learning_rate=args.backbone_finetune_lr
      learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.backbone_finetune_lr,
        decay_steps=len(train_ds) * EPOCHS_FT,
      )
      model.compile(
        optimizer=build_optimizer(args.backbone_optimizer, learning_rate),
        loss=supcon.SupervisedContrastiveLoss(temperature=args.backbone_train_supcon_temperature),
        jit_compile=args.jit_compile,
      )

    if WEIGHTS_EXIST and not args.override:
      print(f"  Previous training found. Skipping training and loading weights from `{WEIGHTS}`.")
    else:
      callbacks = _callbacks(model, "finetune", args) + [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(WEIGHTS, save_weights_only=True, save_best_only=True, verbose=1),
      ]
      model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS_FT, callbacks=callbacks, workers=W, verbose=1)

  if args.step_features_infer:
    inference(model, all_info, distributed_strategy, args, NAME, WEIGHTS, TARGET, W)


def supcon_mh(
    info: "pd.DataFrame",
    all_info: "pd.DataFrame",
    distributed_strategy,
    args,
):
  from core import supcon

  SEED, B_TRAIN, S, P, W, BACKBONE, EPOCHS_HE, EPOCHS_FT, TARGET, NAME, WEIGHTS, WEIGHTS_EXIST = get_extractor_params(args)

  if EPOCHS_HE > 0:
    print(f"SupCon-MultiHead requires training from all layers. Setting backbone_train_epochs = 0 (previous = {EPOCHS_HE}).")
    EPOCHS_HE = 0

  train_info, valid_info = train_test_split(info, test_size=args.backbone_valid_split, stratify=info[TARGET], random_state=SEED)

  TARGETS = ("painter", "style", "genre")
  train_targets = {t: train_info[t] for t in TARGETS}
  valid_targets = {t: valid_info[t] for t in TARGETS}

  train_ds = datasets.pbn.PbNPatchesSequence(train_info, train_targets, B_TRAIN, patch_size=S, patches=P, augment=True)
  valid_ds = datasets.pbn.PbNPatchesSequence(valid_info, valid_targets, B_TRAIN, patch_size=S, patches=P, target_encoder=train_ds.target_encoder)
  C = train_ds.painters
  print(f"painters={len(C)} train={len(train_info)} train_steps={len(train_ds)} valid={len(valid_info)} train_steps={len(valid_ds)}")

  with distributed_strategy.scope():
    backbone_fn = getattr(tf.keras.applications, BACKBONE)
    backbone = backbone_fn(include_top=False, weights="imagenet", input_shape=[S, S, 3], pooling=None)
    model = supcon.supcon_encoder_multitask_network(backbone.input, backbone, name=NAME)

  if EPOCHS_FT > 0:
    print("=" * 65)
    print("Fine-tuning entire model")
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)

    with distributed_strategy.scope():
      unfreeze_top_layers(
        backbone,
        args.backbone_finetune_layers,
        args.backbone_freezebn,
      )

      # learning_rate=args.backbone_finetune_lr
      learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.backbone_finetune_lr,
        decay_steps=len(train_ds) * EPOCHS_FT,
      )
      model.compile(
        optimizer=build_optimizer(args.backbone_optimizer, learning_rate),
        loss={
          'painter': supcon.SupervisedContrastiveLoss(temperature=args.backbone_train_supcon_temperature),
          'style': supcon.SupervisedContrastiveLoss(temperature=args.backbone_train_supcon_temperature),
          'genre': supcon.SupervisedContrastiveLoss(temperature=args.backbone_train_supcon_temperature),
        },
        loss_weights={'painter': 0.4, 'style': 0.3, 'genre': 0.3},
        jit_compile=args.jit_compile,
      )

    if WEIGHTS_EXIST and not args.override:
      print(f"  Previous training found. Skipping training and loading weights from `{WEIGHTS}`.")
    else:
      callbacks = _callbacks(model, "finetune", args) + [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(WEIGHTS, save_weights_only=True, save_best_only=True, verbose=1, monitor="val_project_painter_loss"),
      ]
      model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS_FT, callbacks=callbacks, workers=W, verbose=1)

  if args.step_features_infer:
    inference(model, all_info, distributed_strategy, args, NAME, WEIGHTS, TARGET, W, override=args.override, parts=args.features_parts)


def inference(model, info, distributed_strategy, args, NAME, WEIGHTS, TARGET, workers, override=False, parts=4):
  print("=" * 65)
  print("Inference")

  B = args.batch_test
  P = args.patches_test
  S = args.patch_size
  SPLITS = int(ceil(len(info) / parts))

  FEATURES_LAYER = args.backbone_features_layer
  EXTRACTOR_NAME = f"{NAME}_{FEATURES_LAYER}"

  with distributed_strategy.scope():
    if os.path.exists(WEIGHTS):
      print(f"loading weights from {WEIGHTS}")
      model.load_weights(WEIGHTS, by_name=True)
    else:
      if "pretrained" in NAME:
        print(f"Weights not found at {WEIGHTS}. Ignoring because this model has 'pretrained' in its name. "
              "You can train it by increasing backbone_train_epochs and/or backbone_finetune_epochs.")
      else:
        raise FileNotFoundError(WEIGHTS)

  extractor = tf.keras.Model(model.inputs, model.get_layer(FEATURES_LAYER).output, name=EXTRACTOR_NAME)

  for part in range(parts):
    features_path = os.path.join(args.preds_dir, EXTRACTOR_NAME + f".part-{part}.npz")

    if os.path.exists(features_path) and not override:
      print(f"  features already exist in {features_path} -- skipping inference")
      continue

    info_part = info.iloc[part*SPLITS:(part+1)*SPLITS]
    infer_ds = datasets.pbn.PbNPatchesSequence(info_part, info_part[TARGET], B, patch_size=S, patches=P, augment=False)
    print(f"Predicting {len(infer_ds)} batches (samples={len(info_part)}/{len(info)}, part={part+1}/{parts})", flush=True)

    features = []

    for i in range(len(infer_ds)):
      features_i = extractor.predict_on_batch(infer_ds[i][0])
      features += [features_i]
      if (i+1) % 1000 == 0:
        print(f" {i/len(infer_ds):.0%} done", flush=True)
      elif (i+1) % 100 == 0:
        print(".", end="", flush=True)

    features = np.concatenate(features, axis=0)  # [K, B*P, F]
    features = features.reshape(len(info_part), args.patches_test, *features.shape[1:])

    print(f"  saving features={features.shape} at {features_path}")
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    np.savez_compressed(features_path, filenames=info_part.filename.values, features=features)


def _callbacks(model, tag, args):
  return [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.TensorBoard(os.path.join(args.logs_dir, model.name, tag), histogram_freq=1, write_graph=True, profile_batch=(10, 20)),
  ]


def _metrics():
  return[
    tf.metrics.SparseCategoricalAccuracy(name="top_1"),
    tf.metrics.SparseTopKCategoricalAccuracy(name="top_5"),
  ]

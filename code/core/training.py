import os
import shutil
from typing import List, Optional

import tensorflow as tf

from .utils import log_func_start



def train_fn(
    nn: tf.keras.Model,
    train: tf.data.Dataset,
    valid: tf.data.Dataset,
    epochs: int,
    logs: str,
    weights: str,
    train_steps: Optional[int] = None,
    valid_steps: Optional[int] = None,
    initial_epoch: int = 0,
    override: bool = False,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    distributed_strategy=None,
):
  log_func_start('train_fn', epochs=epochs, logs=logs, weights=weights, train_steps=train_steps, valid_steps=valid_steps, initial_epoch=initial_epoch, override=override)

  try:
    if os.path.exists(logs) and initial_epoch == 0:
      if not override:
        raise ValueError(f'A training was found in {logs}. Either move it or set experiment.override to True.')

      print(f'  Overriding previous training at {logs}.')
      shutil.rmtree(logs)

    os.makedirs(os.path.dirname(weights), exist_ok=True)

    if initial_epoch and os.path.exists(weights):
      print(f'  Restoring previous training from {weights}, fit will restart from epoch {initial_epoch}.')
      if distributed_strategy:
        with distributed_strategy.scope():
          nn.load_weights(weights)
      else:
        nn.load_weights(weights)

    history = nn.fit(
      train,
      validation_data=valid,
      epochs=epochs,
      initial_epoch=initial_epoch,
      callbacks=callbacks,
      steps_per_epoch=train_steps,
      validation_steps=valid_steps,
      verbose=2,
    );

  except KeyboardInterrupt:
    print('\n  interrupted')
    history = nn.history
  else:
    print('\n  done')
  
  return history

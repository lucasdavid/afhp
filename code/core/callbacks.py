import numpy as np
import tensorflow as tf
from tensorflow.keras import backend

tf_logging = tf.get_logger()


class ReduceLRBacktrack(tf.keras.callbacks.ReduceLROnPlateau):
    """
    Reduce Learning Rate on Plateau and restore weights.
    
    Ref: https://stackoverflow.com/a/55228619/2429640
    """
    def __init__(self, best_path, *args, distributed_strategy=None, **kwargs):
      super(ReduceLRBacktrack, self).__init__(*args, **kwargs)
      self.best_path = best_path
      self.distributed_strategy = distributed_strategy

    def on_epoch_end(self, epoch, logs=None):
      current = logs.get(self.monitor)
      if current is None:
        tf_logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                           'which is not available. Available metrics are: %s',
                           self.monitor, ','.join(list(logs.keys())))
        if not self.monitor_op(current, self.best):
          if not self.in_cooldown():
            if self.wait+1 >= self.patience:
              if self.distributed_strategy is None:
                self.restore_weights()
              else:
                with self.distributed_strategy.scope():
                  self.restore_weights()

        super().on_epoch_end(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
      logs = logs or {}
      logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
      current = logs.get(self.monitor)
      if current is None:
        return tf_logging.warning(
          'Learning rate reduction is conditioned on metric `%s` '
          'which is not available. Available metrics are: %s',
          self.monitor, ','.join(list(logs.keys())))

      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

      if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
        return
      
      if not self.in_cooldown():
        self.wait += 1
        if self.wait >= self.patience:
          old_lr = backend.get_value(self.model.optimizer.lr)
          if old_lr > np.float32(self.min_lr):
            new_lr = old_lr * self.factor
            new_lr = max(new_lr, self.min_lr)
            backend.set_value(self.model.optimizer.lr, new_lr)
            
            self.restore_weights()

            if self.verbose > 0:
              print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                    'rate to %s.' % (epoch + 1, new_lr))
            self.cooldown_counter = self.cooldown
            self.wait = 0
  
    def restore_weights(self):
      if self.verbose > 0: print("Backtracking to best model before reducting LR")
      
      if self.distributed_strategy is None:
        self.model.load_weights(self.best_path)
      else:
        with self.distributed_strategy.scope():
          self.model.load_weights(self.best_path)

import tensorflow as tf


def gpus_with_memory_growth():
  gpus = list(tf.config.list_physical_devices('GPU'))

  print(f'Number of devices: {len(gpus)}')

  for d in gpus:
    print(d)
    print(f'  Setting device {d} to memory-growth mode.')
    
    try:
      tf.config.experimental.set_memory_growth(d, True)
    except Exception as e:
      print(e)


def appropriate_distributed_strategy():
  if tf.config.list_physical_devices('GPU'):
    return tf.distribute.MirroredStrategy()
  else:
    return tf.distribute.get_strategy()

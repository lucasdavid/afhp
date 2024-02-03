import random
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def groupby_class(
    names: np.ndarray,
    features: np.ndarray,
    info: "pd.DataFrame",
    class_col: str = "painter",
    verbose: int = 1,
):
  fname2class = dict(info.set_index("filename")[class_col])
  encoder = LabelEncoder()
  classes = np.asarray([fname2class[n] for n in names])
  targets = encoder.fit_transform(classes)

  valid = targets != -1
  features = features[valid]
  targets = targets[valid]

  dataset = {label: features[targets == label] for label in range(len(encoder.classes_))}

  if verbose > 0:
    print(f'labels[:10] = {encoder.classes_[:10]}')
    print(f'samples     = {valid.shape[0]}')
    print(f'valid       = {valid.sum()}')

  return dataset, encoder.classes_


def build_support_query_dataset(
  ds: Dict[int, np.ndarray],
  q_ds: Dict[int, np.ndarray] = None,
  k: int = 10,
  n: int = 5,
  q: int = 1,
  mode: str = 'random',
  virtual_labels: bool = True
):
  """Retrieve a Support Query Dataset Iterator.

  ds:   the dictionary <class, samples> containing feature vectors from the feature extraction network
  q_ds: the query dictionary. If none is passed, we sample from `ds` itself
  k   : number of support samples per class
  n   : number of classes in a task
  q   : number of query samples per class
  mode: the mode in which tasks are retrieved. Options are:
          - 'sequential': labels are drawn in a sequential order
          - 'random': labels are drawn in a random order
  virtual_labels: wether to return virtual or real labels. Virtual labels are in the interval [0, n), and can be directly used
                  in classification loss functions such as tf.nn.sparse_categorical_crossentropy_with_logits.
                  Set it to `False` to return the true labels (with respect to the original dataset's notation) of the samples.
  """
  CLASS_NAMES = list(ds.keys())

  # get patch number from dictionary:
  # {van-gogh: <tensor shape=(N_PAINTS, N_PATCHES, N_FEATS)>, ...}
  N_PATCHES = ds[CLASS_NAMES[0]].shape[1]

  if q_ds and not ds.keys() <= q_ds.keys():
    raise ValueError(f'The query set {q_ds.keys()} must contain all labels in the support set {ds.keys()}.')

  while True:
    if mode == 'random':
      random.shuffle(CLASS_NAMES)
    
    for c in range(0, len(CLASS_NAMES) - n + 1, n):
      ss_ = []
      qs_ = []

      task_labels = CLASS_NAMES[c:c+n]

      for label in task_labels:
        if q_ds is None:
          ss = ds[label][np.random.choice(len(ds[label]), k + q)]
          ss, qs = ss[:k], ss[k:]
        else:
          ss = ds[label][np.random.choice(len(ds[label]), k)]
          qs = q_ds[label][np.random.choice(len(q_ds[label]), q)]

        if N_PATCHES > 1:
          # Paintings have more than one patch. Take one at random.
          ss = ss[range(k), np.random.choice(N_PATCHES, size=k)]
          qs = qs[range(q), np.random.choice(N_PATCHES, size=q)]
        else:
          ss, qs = ss[:, 0], qs[:, 0]  # Select only patch available.

        ss_.append(ss)
        qs_.append(qs)

      ss_ = np.asarray(ss_)
      qs_ = np.concatenate(qs_)

      if virtual_labels:
        # Within-context labels.
        y = np.arange(len(task_labels), dtype=np.int32)
      else:
        y = task_labels

      yq = np.repeat(y, q)

      yield ss_, y, qs_, yq


def build_support_query_tf_dataset(samples_map, k, n, q, f, mode):
  def _data_gen():
    for s in build_support_query_dataset(samples_map, k=k, n=n, q=q, mode=mode):
      yield s

  signature = (
    tf.TensorSpec(shape=(n, k, f), dtype=tf.float32),
    tf.TensorSpec(shape=(n,), dtype=tf.int32),
    tf.TensorSpec(shape=(n*q, f), dtype=tf.float32),
    tf.TensorSpec(shape=(n*q,), dtype=tf.int32),
  )

  return tf.data.Dataset.from_generator(_data_gen, output_signature=signature)


class PbNFeaturesSequence(tf.keras.utils.Sequence):
  def __init__(self, info, features, targets, batch_size, sample_size=None, patches=1, target_encoder=None, augment=False):
    self.info = info
    self.features = features
    self.batch_size = batch_size
    self.sample_size = sample_size
    self.patches = patches
    self.augment = augment

    if not isinstance(targets, dict):
      self.multi_output = False
      targets = {"group_0": targets}
      if target_encoder is not None and not isinstance(target_encoder, dict):
        target_encoder = {"group_0": target_encoder}
    else:
      self.multi_output = True

    self.targets = targets
    self.target_encoder = target_encoder or {g: LabelEncoder().fit(y) for g, y in targets.items()}
    self.target_indices = {g: self.target_encoder[g].transform(y) for g, y in targets.items()}

  @property
  def painters(self):
    key = "painter" if "painter" in self.target_encoder else "group_0"
    return self.target_encoder[key].classes_

  def __len__(self):
    return math.ceil(len(self.info) / self.batch_size)

  def __getitem__(self, idx):
    low = idx * self.batch_size
    high = min(low + self.batch_size, len(self.info))

    batch_images = []

    for p in self.info[low:high].full_path:
      batch_images.append(self.transform(img))

    batch_images = np.stack(batch_images, 0)

    if self.multi_output:
      targets = {name: np.repeat(tensor[low:high].astype("int64"), self.patches)
                  for name, tensor in self.target_indices.items()}
    else:
      targets = np.repeat(self.target_indices["group_0"][low:high].astype("int64"), self.patches)

    return batch_images, targets


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
  from sklearn.manifold import TSNE

  with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    z = TSNE(init='pca', learning_rate='auto').fit_transform(x)
    y = np.char.add("class ", y.astype("str"))

    sns.scatterplot(x=z[is_real, 0], y=z[is_real, 1], hue=y[is_real], marker='s', label='Original', alpha=0.8, legend=False)
    sns.scatterplot(x=z[~is_real, 0], y=z[~is_real, 1], hue=y[~is_real], label='Generated', alpha=0.8, legend='brief')
    plt.legend()

# endregion

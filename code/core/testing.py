import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import metrics as skmetrics

import matplotlib.pyplot as plt
import seaborn as sns


def metrics_per_label(gt, probs, threshold=0.5, classes=None):
  p_pred = tf.cast(probs > threshold, probs.dtype).numpy()

  tru_ = tf.reduce_sum(gt, axis=0)
  neg_ = tf.reduce_sum(1- gt, axis=0)

  tpr = tf.reduce_sum(p_pred*gt, axis=0) / tru_
  fpr = tf.reduce_sum(p_pred*(1-gt), axis=0) / neg_
  tnr = tf.reduce_sum((1-p_pred)*(1-gt), axis=0) / neg_
  fnr = tf.reduce_sum((1-p_pred)*gt, axis=0) / tru_

  f2_score = skmetrics.fbeta_score(gt, p_pred, beta=2, average=None)
  precision, recall, f1_score, support = skmetrics.precision_recall_fscore_support(
    gt, p_pred, average=None)

  mcm = skmetrics.multilabel_confusion_matrix(gt, p_pred)

  skmetrics.accuracy_score(gt, p_pred)

  try: roc_auc = skmetrics.roc_auc_score(gt, probs, average=None)
  except ValueError: roc_auc = None

  return pd.DataFrame({
    'true positive r': tpr,
    'true negative r': tnr,
    'false positive r': fpr,
    'false negative r': fnr,
    'precision': precision,
    'recall': recall,
    'auc_score': roc_auc,
    'f1_score': f1_score,
    'f2_score': f2_score,
    'support': support,
    'label': classes
  }), mcm


def labels_and_probs(nn, dataset):
  labels_ = []
  probs_ = []

  for ix, (images, labels) in enumerate(dataset):
    y = nn(images, training=False)
    y = tf.nn.sigmoid(y)

    labels_.append(labels)
    probs_.append(y)

    print('.', end='' if (ix+1) % 120 else '\n')
  
  return (tf.concat(labels_, axis=0),
          tf.concat(probs_, axis=0))


def plot_occurrence_and_prediction_matrices(
    l,
    p,
    threshold=0.5,
    classes=None,
    to_file='occur_pred.png'
):
  co_occurrence = tf.transpose(l) @ l
  co_occurrence_rate = tf.math.divide_no_nan(
    co_occurrence,
    tf.reshape(np.diag(co_occurrence), (-1, 1)))

  d = tf.cast(p > threshold, p.dtype)
  co_prediction = tf.transpose(d) @ d
  co_prediction_rate = tf.math.divide_no_nan(
    co_prediction,
    tf.reshape(np.diag(co_prediction), (-1, 1)))

  plt.figure(figsize=(24, 12))
  plt.subplot(131)
  plt.title('Co-occurrence Rates')
  sns.heatmap(
    co_occurrence_rate.numpy(),
    annot=True,
    fmt='.0%',
    xticklabels=classes,
    yticklabels=classes,
    cmap="RdPu",
    cbar=False
  )
  
  plt.subplot(132)
  plt.title('Co-prediction Rates')
  sns.heatmap(
    co_prediction_rate.numpy(),
    annot=True,
    fmt='.0%',
    xticklabels=classes,
    yticklabels=classes,
    cmap="RdPu",
    cbar=False
  )
  
  plt.subplot(133)
  plt.title('Difference')
  sns.heatmap(
    tf.abs(co_occurrence_rate - co_prediction_rate).numpy(),
    annot=True,
    fmt='.0%',
    xticklabels=classes,
    yticklabels=classes,
    cmap="RdPu",
    cbar=False
  )

  plt.tight_layout()

  if to_file:
    plt.savefig(to_file);

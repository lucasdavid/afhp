import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics as skmetrics

from core.utils import get_afhp_params
from core.fsl import build_support_query_dataset


def run(args, afhp_model, c_samples, classes):
  _, _, _, _, NOISE, _, _, _, _, _ = get_afhp_params(args)

  N_WAY = [5]
  K_SHOT  = [1, 5, 10]
  EXTRA_K = [0, 20, 60, 100, 140, 300]
  q = 10
  repeat = 10
  reports = []

  try:
    for n in N_WAY:
      for k in K_SHOT:
        for extra_k in EXTRA_K:
          results = _run_experiment(afhp_model, c_samples, classes, n, k, q, extra_k, repeat, NOISE)
          reports.append((n, k, q, extra_k, *results))

  except KeyboardInterrupt:
    print('\ninterrupted')

  cols = 'n k q extra tasks loss acc bal_acc auc'.split()
  reports = pd.DataFrame(reports, columns=cols)

  print("Evaluation Report")
  print(reports.round(3))
  print("-" * 80)

  save_path =  os.path.join(args.logs_dir, afhp_model.name, "fsl-report.csv")
  reports.to_csv(save_path)

  return reports


# region Few-Shot Learning Classification

def _run_experiment(
    afhp_model: "AFHP",
    c_samples,
    classes,
    n_way,
    k_shot,
    q_test,
    extra_k,
    repeat,
    NOISE,
):
  test_ds = build_support_query_dataset(c_samples, k=k_shot, n=n_way, q=q_test)
  tasks = repeat * len(classes) // n_way

  all_results = np.zeros(4, dtype="float32")

  print(f"FSL Experiment way={n_way} shot={k_shot} query={q_test} extra={extra_k}:", end=" ")

  for task, (x_real, y_real, x_query, y_query) in enumerate(test_ds):
    names = ("x_real", "y_real", "x_query", "y_query")
    vectors = (x_real, y_real, x_query, y_query)
    isinf = [np.isinf(x).any() for x in vectors]

    for n, a, _isinf in zip(names, vectors, isinf):
      if _isinf:
        raise ValueError(
          f"Array {n} (shape={a.shape} dtype={a.dtype}) has inf values (min={a.min()} max={a.max()} mean={a.mean()}):" + str(a.round(1))  
        )

    context = tf.reduce_mean(x_real, axis=1) # [5,10,2048]

    x = tf.reshape(x_real, (k_shot*n_way, x_real.shape[-1]))
    y = tf.repeat(y_real, k_shot, axis=0)

    if extra_k:
      # Hallucinate GC new samples using the generator.
      context = tf.repeat(context, extra_k, axis=0)
      z1 = tf.random.normal((context.shape[0], NOISE))

      x1_fake = afhp_model.G.predict_on_batch([z1, context])
      
      y_fake = tf.repeat(y_real, extra_k, axis=0)

      x_real = tf.reshape(x_real, (k_shot*n_way, x_real.shape[-1]))
      y_real = tf.repeat(y_real, k_shot, axis=0)

      x, y = (tf.concat((x, x1_fake), axis=0),
              tf.concat((y, y_fake), axis=0))
    
    x = tf.cast(x, tf.float32).numpy()
    y = tf.cast(y, tf.int32).numpy()
    x_query = x_query.astype("float32")
    y_query = y_query.astype("int32")

    batch_results = _train_eval_fsl_classifier(x, y, x_query, y_query)
    all_results += batch_results

    if (task+1) % 100 == 0:
      print(f"{(task+1)}/{tasks} ({(task+1)/tasks:.0%})", end=" ")
    elif (task+1) % 10 == 0:
      print(".", end="")

    del x, y, x_query, y_query

    if task >= tasks:
      break
  
  print()

  all_results /= task

  return (task, *all_results.tolist())

def _train_eval_fsl_classifier(x, y, x_query, y_query):
  classifier = make_pipeline(StandardScaler(), SVC(probability=True))
  classifier.fit(x, y)

  y_pred = classifier.predict(x_query)
  y_score = classifier.predict_proba(x_query)

  lloss = -np.log(y_score[range(len(x_query)), y_query.flat]).sum()

  acc = skmetrics.accuracy_score(y_query, y_pred)
  bacc = skmetrics.balanced_accuracy_score(y_query, y_pred)
  auc = skmetrics.roc_auc_score(y_query, y_score, average="macro", multi_class="ovr")
  # p, r, f1, sup = skmetrics.precision_recall_fscore_support(y_query, y_pred, average="macro")

  return np.asarray([lloss, acc, bacc, auc], dtype="float32")

# endregion

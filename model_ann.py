import warnings
warnings.filterwarnings("ignore")

import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
  classification_report,
  confusion_matrix,
  precision_score,
  recall_score,
  fbeta_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from base_functions import (
  RANDOM_STATE,
  get_data_and_features,
  get_preprocessor,
  main_template,
)


def encontrar_melhor_threshold(y_true, y_proba, beta=2.0):
  """
  Encontra o threshold que maximiza o F-beta (beta > 1 dá mais peso ao recall).
  Também retorna o recall e a precision nesse threshold.
  """
  thresholds = np.linspace(0.1, 0.9, 81)

  best_thr = 0.5
  best_fbeta = -1.0
  best_recall = 0.0
  best_precision = 0.0

  for thr in thresholds:
    y_pred = (y_proba >= thr).astype(int)

    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    # critério: maximizar F-beta
    if fbeta > best_fbeta:
      best_fbeta = fbeta
      best_thr = thr
      best_recall = rec
      best_precision = prec

  return best_thr, best_fbeta, best_recall, best_precision


def treinar_modelo(df, model_name):
  X, y, num_cols, cat_cols = get_data_and_features(df)
  pre = get_preprocessor(X, num_cols, cat_cols)

  clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=RANDOM_STATE,
    early_stopping=True,
  )

  pipe = Pipeline(steps=[
    ("preprocess", pre),
    ("clf", clf),
  ])

  cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
  )

  scoring = {
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
  }

  results = cross_validate(
    pipe,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False,
  )

  # ===== Probabilidades out-of-fold para ajuste de threshold =====
  print("\n[ANN] Gerando probabilidades com cross_val_predict para ajustar threshold (otimizar F2)...")
  y_proba = cross_val_predict(
    pipe,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
  )[:, 1]  # probabilidade da classe 1 (rejeição)

  # ===== Encontrar melhor threshold com foco em recall/F2 =====
  best_thr, best_f2, best_recall_thr, best_precision_thr = encontrar_melhor_threshold(
    y_true=y,
    y_proba=y_proba,
    beta=2.0,
  )

  print(f"\n[ANN] Melhor threshold (F2): {best_thr:.3f}")
  print(f"[ANN] Recall (classe 1) no melhor threshold: {best_recall_thr:.4f}")
  print(f"[ANN] Precision (classe 1) no melhor threshold: {best_precision_thr:.4f}")
  print(f"[ANN] F2-score no melhor threshold: {best_f2:.4f}")

  # ===== Matriz de confusão usando o melhor threshold =====
  y_pred_best = (y_proba >= best_thr).astype(int)
  cm = confusion_matrix(y, y_pred_best)
  tn, fp, fn, tp = cm.ravel()

  print("\n[ANN] Matriz de confusão (usando melhor threshold):")
  print(cm)
  print("\nFN (falsos negativos = rejeição não detectada) é o erro mais crítico!")

  print("\n[ANN] Relatório de classificação (melhor threshold):\n")
  print(classification_report(
    y,
    y_pred_best,
    digits=4,
    target_names=["Sem Rejeição (0)", "Com Rejeição (1)"],
  ))

  metrics = {
    "model": model_name,
    "roc_auc_mean": np.mean(results["test_roc_auc"]),
    "roc_auc_std": np.std(results["test_roc_auc"]),
    "recall_mean": np.mean(results["test_recall"]),
    "recall_std": np.std(results["test_recall"]),
    "precision_mean": np.mean(results["test_precision"]),
    "precision_std": np.std(results["test_precision"]),
    "f1_mean": np.mean(results["test_f1"]),
    "f1_std": np.std(results["test_f1"]),
    "accuracy_mean": np.mean(results["test_accuracy"]),
    "accuracy_std": np.std(results["test_accuracy"]),

    # infos específicas do threshold escolhido
    "best_threshold": float(best_thr),
    "recall_at_best_threshold": float(best_recall_thr),
    "precision_at_best_threshold": float(best_precision_thr),
    "f2_at_best_threshold": float(best_f2),

    # matriz de confusão no melhor threshold
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
  }

  return metrics


if __name__ == "__main__":
  main_template(treinar_modelo, "Artificial Neural Networks (Cross-Validation)")
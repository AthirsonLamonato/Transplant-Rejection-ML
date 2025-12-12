import warnings
import os
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
  classification_report,
  confusion_matrix,
  precision_score,
  recall_score,
  fbeta_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from base_functions import (
  RANDOM_STATE,
  CLASS_WEIGHT,
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
  # ===== Pré-processamento =====
  X, y, num_cols, cat_cols = get_data_and_features(df)
  pre = get_preprocessor(X, num_cols, cat_cols)

  # ===== Modelo =====
  clf = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    class_weight=CLASS_WEIGHT,
    max_depth=15,
    min_samples_leaf=50,
  )

  pipe = Pipeline(steps=[
    ("preprocess", pre),
    ("clf", clf),
  ])

  # ===== Cross-validation com métricas padrão =====
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
  print("\n[Decision Tree] Gerando probabilidades com cross_val_predict para ajustar threshold (otimizar F2)...")
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

  print(f"\n[Decision Tree] Melhor threshold (F2): {best_thr:.3f}")
  print(f"[Decision Tree] Recall (classe 1) no melhor threshold: {best_recall_thr:.4f}")
  print(f"[Decision Tree] Precision (classe 1) no melhor threshold: {best_precision_thr:.4f}")
  print(f"[Decision Tree] F2-score no melhor threshold: {best_f2:.4f}")

  # ===== Matriz de confusão usando o melhor threshold =====
  y_pred_best = (y_proba >= best_thr).astype(int)
  cm = confusion_matrix(y, y_pred_best)
  tn, fp, fn, tp = cm.ravel()

  print("\n[Decision Tree] Matriz de confusão (usando melhor threshold):")
  print(cm)
  print("\nFN (falsos negativos = rejeição não detectada) é o erro mais crítico!")

  print("\n[Decision Tree] Relatório de classificação (melhor threshold):\n")
  print(classification_report(
    y,
    y_pred_best,
    digits=4,
    target_names=["Sem Rejeição (0)", "Com Rejeição (1)"],
  ))

  # ===== Importância de features =====
  try:
    print("\n[Decision Tree] Ajustando modelo final em todo o dataset para calcular importâncias de features...")
    final_pipe = pipe.fit(X, y)

    model_step = final_pipe.named_steps["clf"]
    preprocessor = final_pipe.named_steps["preprocess"]

    if hasattr(model_step, "feature_importances_"):
      importances = model_step.feature_importances_

      try:
        raw_feature_names = preprocessor.get_feature_names_out()
      except Exception:
        raw_feature_names = np.array([f"feat_{i}" for i in range(len(importances))])

      if len(raw_feature_names) != len(importances):
        raw_feature_names = np.array([f"feat_{i}" for i in range(len(importances))])

      clean_names = []
      feature_type = []
      for fname in raw_feature_names:
        if fname.startswith("num__"):
          feature_type.append("numérica")
          clean_names.append(fname.replace("num__", "", 1))
        elif fname.startswith("cat__"):
          feature_type.append("categórica")
          clean_names.append(fname.replace("cat__", "", 1))
        else:
          feature_type.append("desconhecida")
          clean_names.append(fname)

      importance_pct = (importances / importances.sum()) * 100.0

      imp_df = (
        pd.DataFrame({
          "feature": clean_names,
          "feature_type": feature_type,
          "importance_pct": importance_pct,
        })
        .sort_values("importance_pct", ascending=False)
      )

      top_n = min(10, len(imp_df))
      print(f"\n[Decision Tree] Top {top_n} variáveis mais importantes (em %):")
      print(imp_df.head(top_n).to_string(index=False))

      base_name = model_name.lower().replace(' ', '_')

      os.makedirs("Features", exist_ok=True)
      outname = os.path.join("Features", f"feature_importances_{base_name}.csv")
      imp_df.to_csv(outname, index=False)
      print(f"\n[Decision Tree] Importâncias salvas em {outname}")

      # Gera gráfico de importâncias
      try:
        os.makedirs("Gráficos", exist_ok=True)
        top_plot = min(10, len(imp_df))
        to_plot = imp_df.head(top_plot).iloc[::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(to_plot["feature"], to_plot["importance_pct"], height=0.4)
        plt.xlabel("Importância (%)")
        plt.title(f"Top {top_plot} variáveis mais importantes — {model_name}")
        plt.tight_layout()
        outname_png = os.path.join("Gráficos", f"feature_importances_{base_name}.png")
        plt.savefig(outname_png, dpi=300)
        plt.close()
        print(f"[Decision Tree] Gráfico de importâncias salvo em {outname_png}")
      except Exception as e:
        print(f"[AVISO] Não foi possível gerar gráfico de importâncias para {model_name}: {e}")
    else:
      print("[AVISO] Modelo DecisionTree não possui atributo feature_importances_.")
  except Exception as e:
    print(f"[AVISO] Não foi possível calcular importâncias de features para {model_name}: {e}")

  # ===== Métricas agregadas =====
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
  main_template(treinar_modelo, "Decision Tree (Cross-Validation)")
import warnings
import os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
  classification_report,
  confusion_matrix,
  precision_score,
  recall_score,
  fbeta_score,
)

from xgboost import XGBClassifier

from base_functions import (
  RANDOM_STATE,
  CLASS_WEIGHT,
  get_data_and_features,
  get_preprocessor,
  main_template,
)


def encontrar_melhor_threshold(y_true, y_proba, beta=2.0):
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

  # ===== Cálculo do peso da classe positiva para o XGBoost =====
  n_pos = y.sum()
  n_neg = len(y) - n_pos

  if n_pos == 0 or n_neg == 0:
    # caso extremo, não dá pra calcular razão, evita divisão por zero
    scale_pos_weight = 1.0
    print("[AVISO] Classe completamente desbalanceada (só 0 ou só 1). "
          "Usando scale_pos_weight = 1.0.")
  else:
    scale_pos_weight = n_neg / n_pos
    print(f"[XGBoost] scale_pos_weight calculado como: {scale_pos_weight:.4f} (negativos/positivos)")

  # ===== Modelo XGBoost =====
  clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
  )

  from sklearn.pipeline import Pipeline
  pipe = Pipeline(steps=[
    ("preprocess", pre),
    ("clf", clf),
  ])

  # ===== Cross-validation =====
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

  # ===== Probabilidades out-of-fold =====
  print("\n[XGBoost] Gerando probabilidades out-of-fold para achar melhor threshold...")
  y_proba = cross_val_predict(
    pipe,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
  )[:, 1]

  # ===== Threshold =====
  best_thr, best_f2, best_recall_thr, best_precision_thr = encontrar_melhor_threshold(
    y_true=y,
    y_proba=y_proba,
    beta=2.0,
  )

  print(f"\n[XGBoost] Melhor threshold: {best_thr:.4f}")
  print(f"[XGBoost] F2 no melhor threshold: {best_f2:.4f}")
  print(f"[XGBoost] Recall no melhor threshold: {best_recall_thr:.4f}")
  print(f"[XGBoost] Precisão no melhor threshold: {best_precision_thr:.4f}")

  y_pred_best = (y_proba >= best_thr).astype(int)

  # ===== Matriz de confusão =====
  cm = confusion_matrix(y, y_pred_best)
  tn, fp, fn, tp = cm.ravel()

  print("\n[XGBoost] Matriz de confusão (melhor threshold):")
  print(cm)
  print("\nFN (falsos negativos) são os mais críticos!")

  print("\n[XGBoost] Relatório de classificação (melhor threshold):\n")
  print(classification_report(
    y,
    y_pred_best,
    digits=4,
    target_names=["Sem Rejeição (0)", "Com Rejeição (1)"],
  ))

  # ===== Importância de features =====
  try:
    print("\n[XGBoost] Ajustando modelo final em todo o dataset para calcular importâncias de features...")
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
      print(f"\n[XGBoost] Top {top_n} variáveis mais importantes (em %):")
      print(imp_df.head(top_n).to_string(index=False))

      base_name = model_name.lower().replace(' ', '_')

      os.makedirs("Features", exist_ok=True)
      outname_csv = os.path.join("Features", f"feature_importances_{base_name}.csv")
      imp_df.to_csv(outname_csv, index=False)
      print(f"\n[XGBoost] Importâncias salvas em {outname_csv}")

      # gráfico
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
        print(f"[XGBoost] Gráfico de importâncias salvo em {outname_png}")
      except Exception as e:
        print(f"[AVISO] Não foi possível gerar gráfico de importâncias para {model_name}: {e}")
    else:
      print("[AVISO] Modelo XGBoost não possui atributo feature_importances_.")
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

    "best_threshold": float(best_thr),
    "recall_at_best_threshold": float(best_recall_thr),
    "precision_at_best_threshold": float(best_precision_thr),
    "f2_at_best_threshold": float(best_f2),

    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
  }

  return metrics


if __name__ == "__main__":
  main_template(treinar_modelo, "XGBoost (Cross-Validation)")
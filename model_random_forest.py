import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from base_functions import (RANDOM_STATE, CLASS_WEIGHT, get_data_and_features, get_preprocessor, main_template)

def treinar_modelo(df, model_name):
  X, y, num_cols, cat_cols = get_data_and_features(df)
  pre = get_preprocessor(X, num_cols, cat_cols)

  clf = Pipeline(
    steps=[
      ("prep", pre),
      ("model", RandomForestClassifier(random_state=RANDOM_STATE, class_weight=CLASS_WEIGHT, n_estimators=200, max_depth=15, min_samples_leaf=20,n_jobs=-1)),
    ]
  )

  print("\nExecutando Validação Cruzada Estratificada (5 Folds)...")
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

  scoring = {
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
  }

  results = cross_validate(
    clf,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1,
  )

  print("\n==================================================")
  print(f"     MÉTRICAS MÉDIAS - {model_name} (5 Folds)     ")
  print("==================================================")
  for metric, values in results.items():
    if metric.startswith("test_"):
      name = metric.replace("test_", "").upper()
      print(f"{name:10s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

  print("==================================================")

  print("\nGerando predições com cross_val_predict para matriz de confusão...")
  y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

  cm = confusion_matrix(y, y_pred)
  tn, fp, fn, tp = cm.ravel()

  print("\nMATRIZ DE CONFUSÃO (validação cruzada agregada):")
  print(cm)

  print("\nRelatório de classificação (usando predições da validação cruzada):\n")
  print(classification_report(y, y_pred, digits=4, target_names=["Sem Rejeição (0)", "Com Rejeição (1)"]))

  print("\nTreinando modelo final em todos os dados...")
  clf.fit(X, y)

  try:
    if hasattr(clf.named_steps["model"], "feature_importances_"):
      importances = clf.named_steps["model"].feature_importances_
      preprocessor = clf.named_steps["prep"]
      feature_names = list(preprocessor.get_feature_names_out())

      imp_df = (
        pd.DataFrame({"feature": feature_names, "importance_pct": importances * 100})
        .sort_values("importance_pct", ascending=False)
      )
      imp_df["importance_pct"] = imp_df["importance_pct"].round(4)

      top_n = min(30, len(imp_df))
      print(f"\nTop {top_n} variáveis mais importantes (em %):")
      print(imp_df.head(top_n).to_string(index=False))

      out_name = f"feature_importances_{model_name.lower().replace(' ', '_')}.csv"
      imp_df.to_csv(out_name, index=False)
      print(f"\n✓ Importâncias salvas em {out_name}")
  except Exception as e:
    print(f"[AVISO] Não foi possível calcular importâncias: {e}")

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
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
  }

  return metrics

if __name__ == "__main__":
  main_template(treinar_modelo, "Random Forest (Cross-Validation)")

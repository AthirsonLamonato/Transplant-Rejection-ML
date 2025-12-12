import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base_functions import carregar_dados_com_subamostragem, tratar_features, RANDOM_STATE, CLASS_WEIGHT
from model_decision_tree import treinar_modelo as treinar_dt
from model_random_forest import treinar_modelo as treinar_rf
from model_xgboost import treinar_modelo as treinar_xgb
from model_ann import treinar_modelo as treinar_ann

def main():
  print("Carregando e processando dados uma única vez...")
  df = carregar_dados_com_subamostragem()
  print("\nTratando features...")
  df = tratar_features(df)

  metrics_list = []

  print("\n===== Decision Tree =====")
  metrics_list.append(treinar_dt(df, "Decision Tree (Cross-Validation)"))

  print("\n===== Random Forest =====")
  metrics_list.append(treinar_rf(df, "Random Forest (Cross-Validation)"))

  print("\n===== Artificial Neural Networks =====")
  metrics_list.append(treinar_ann(df, "Artificial Neural Networks (Cross-Validation)"))

  print("\n===== XGBoost =====")
  metrics_list.append(treinar_xgb(df, "XGBoost (Cross-Validation)"))

  # Remove entradas None (caso algum modelo falhe)
  metrics_list = [m for m in metrics_list if m is not None]

  df_metrics = pd.DataFrame(metrics_list)
  os.makedirs("Comparações", exist_ok=True)
  out_name = os.path.join("Comparações", "comparacao_modelos.csv")
  df_metrics.to_csv(out_name, index=False)

  print(f"\nCSV de comparação salvo como: {out_name}\n")
  print(df_metrics)

  # =======================
  # EXPLICAÇÕES DAS MÉTRICAS
  # =======================
  print("\n" + "=" * 80)
  print("EXPLICAÇÃO DAS MÉTRICAS (para interpretação dos resultados)")
  print("=" * 80)

  explicacoes = {
    "model": "Nome do modelo avaliado (ex: Decision Tree, Random Forest, ANN).",
    "roc_auc_mean": "Média da métrica ROC-AUC — mede a capacidade do modelo de distinguir entre rejeição e não rejeição.",
    "roc_auc_std": "Desvio padrão do ROC-AUC — mostra o quanto o resultado variou entre os folds.",
    "recall_mean": "Média do Recall (Sensibilidade) — percentual de rejeições corretamente detectadas.",
    "recall_std": "Desvio padrão do Recall entre os folds.",
    "precision_mean": "Média da Precisão — percentual de previsões de rejeição que estavam realmente corretas.",
    "precision_std": "Desvio padrão da Precisão entre os folds.",
    "f1_mean": "Média do F1-Score — equilíbrio entre Precisão e Recall.",
    "f1_std": "Desvio padrão do F1-Score entre os folds.",
    "accuracy_mean": "Média da Acurácia — percentual geral de acertos.",
    "accuracy_std": "Desvio padrão da Acurácia entre os folds.",
    "tn": "True Negatives — casos corretamente previstos como 'sem rejeição'.",
    "fp": "False Positives — casos sem rejeição, mas previstos incorretamente como rejeição (falsos alarmes).",
    "fn": "False Negatives — casos com rejeição que o modelo não detectou (erro mais crítico).",
    "tp": "True Positives — casos corretamente previstos como rejeição.",
    "best_threshold": "Limiar de decisão que maximizou o F2 para o modelo (quando calculado).",
    "recall_at_best_threshold": "Recall da classe positiva usando o melhor limiar (F2).",
    "precision_at_best_threshold": "Precisão da classe positiva usando o melhor limiar (F2).",
    "f2_at_best_threshold": "Valor do F2-Score no melhor limiar.",
  }

  for k, v in explicacoes.items():
    print(f"{k:<25} => {v}")

  print("=" * 80)

  # =======================
  # EXPLICAÇÃO DOS PARÂMETROS DOS MODELOS (COM VALORES)
  # =======================
  print("\n" + "=" * 80)
  print("EXPLICAÇÃO DOS PRINCIPAIS PARÂMETROS DOS MODELOS (COM VALORES DEFINIDOS)")
  print("=" * 80)

  parametros = {
    "Decision Tree (Árvore de Decisão)": {
      "random_state": f"{RANDOM_STATE} => Garante que o resultado seja reproduzível em todas as execuções.",
      "class_weight": f"'{CLASS_WEIGHT}' => Balanceia o peso entre as classes (dá mais importância aos casos de rejeição).",
      "max_depth": "15 => Limita a profundidade da árvore para evitar overfitting (decorar os dados).",
      "min_samples_leaf": "50 => Exige ao menos 50 amostras por folha, tornando as decisões mais robustas."
    },
    "Random Forest (Floresta Aleatória)": {
      "random_state": f"{RANDOM_STATE} => Mantém os resultados consistentes entre execuções.",
      "class_weight": f"'{CLASS_WEIGHT}' => Compensa o desbalanceamento entre rejeição e não rejeição.",
      "n_estimators": "200 => Número de árvores dentro da floresta (mais árvores = mais estabilidade).",
      "max_depth": "15 => Limita o tamanho de cada árvore, controlando o overfitting.",
      "min_samples_leaf": "20 => Define o mínimo de amostras por folha, evitando regras baseadas em poucos exemplos.",
      "n_jobs": "-1 => Usa todos os núcleos do processador para acelerar o treinamento."
    },
    "Artificial Neural Network (MLPClassifier)": {
      "random_state": f"{RANDOM_STATE} => Garante reprodutibilidade dos pesos iniciais da rede.",
      "hidden_layer_sizes": "(100, 50) => Define duas camadas ocultas com 100 e 50 neurônios respectivamente.",
      "max_iter": "500 => Limite máximo de iterações de treino (quantas vezes a rede ajusta os pesos).",
      "early_stopping": "True => Interrompe o treino automaticamente se o modelo parar de melhorar (evita overfitting).",
      "n_iter_no_change": "20 => Número de iterações sem melhora toleradas antes de parar o treino."
    }
  }

  for modelo, params in parametros.items():
    print(f"\n{modelo}")
    for k, v in params.items():
      print(f"{k:<20} => {v}")

  print("=" * 80)

  # =======================
  # HEATMAP – MODELOS x MÉTRICAS (0.5 E THRESHOLD OTIMIZADO)
  # =======================
  print("\nGerando heatmap...")

  if {"recall_at_best_threshold", "precision_at_best_threshold", "f2_at_best_threshold"}.issubset(df_metrics.columns):
    # Heatmap com métricas de 0.5 e do melhor threshold
    metrics_heatmap = [
      "roc_auc_mean",              # indep. de threshold, mas importante no painel
      "accuracy_mean",             # idem
      "recall_mean",               # threshold 0.5
      "precision_mean",            # threshold 0.5
      "f1_mean",                   # threshold 0.5
      "recall_at_best_threshold",  # threshold otimizado
      "precision_at_best_threshold",
      "f2_at_best_threshold",
    ]
    metric_labels_heatmap = [
      "ROC-AUC (CV)",
      "Acurácia (CV)",
      "Recall (0.5)",
      "Precisão (0.5)",
      "F1-Score (0.5)",
      "Recall (thr otimizado)",
      "Precisão (thr otimizado)",
      "F2 (thr otimizado)",
    ]
  else:
    # Caso não haja colunas de threshold otimizado, faz um heatmap básico apenas com 0.5
    metrics_heatmap = ["roc_auc_mean", "recall_mean", "precision_mean", "f1_mean", "accuracy_mean"]
    metric_labels_heatmap = ["ROC-AUC", "Recall (0.5)", "Precisão (0.5)", "F1-Score (0.5)", "Acurácia (CV)"]

  data_heatmap = df_metrics[metrics_heatmap].to_numpy()

  fig, ax = plt.subplots(figsize=(12, 6))
  img = ax.imshow(data_heatmap, aspect="auto", cmap='YlGnBu')

  ax.set_yticks(np.arange(len(df_metrics)))
  ax.set_yticklabels(df_metrics["model"])

  ax.set_xticks(np.arange(len(metrics_heatmap)))
  ax.set_xticklabels(metric_labels_heatmap, rotation=15, ha='right')

  # valores dentro das células
  for i in range(data_heatmap.shape[0]):
    for j in range(data_heatmap.shape[1]):
      ax.text(j, i, f"{data_heatmap[i, j]:.3f}",
              ha="center", va="center", fontsize=9, color="black")

  ax.set_title("Heatmap – Modelos x Métricas (Threshold 0.5 e Otimizado)")
  fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

  plt.tight_layout()
  out_heatmap = os.path.join("Comparações", "heatmap_modelos_metricas.png")
  plt.savefig(out_heatmap, dpi=300)
  print(f"Heatmap salvo como: {out_heatmap}")

  plt.close()

  # =======================
  # GRÁFICO PRINCIPAL – BARRAS AGRUPADAS (SOMENTE THRESHOLD 0.5)
  # =======================
  print("\nGerando gráfico de comparação com threshold padrão (0.5)...")

  # Comparação entre modelos APENAS com threshold padrão
  metrics_to_plot = [
    "roc_auc_mean",
    "accuracy_mean",
    "recall_mean",
    "precision_mean",
    "f1_mean",
  ]
  metric_labels = [
    "ROC-AUC (CV)",
    "Acurácia (CV)",
    "Recall (0.5)",
    "Precisão (0.5)",
    "F1-Score (0.5)",
  ]

  x = np.arange(len(metrics_to_plot))
  n_models = len(df_metrics)
  width = 0.15
  offset_center = (n_models - 1) / 2

  fig, ax = plt.subplots(figsize=(14, 7))

  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

  for i, row in df_metrics.iterrows():
    offset = (i - offset_center) * width
    values = [row[m] for m in metrics_to_plot]
    ax.bar(x + offset, values, width, label=row["model"], color=colors[i])

  ax.set_xticks(x)
  ax.set_xticklabels(metric_labels)
  ax.set_ylim(0.0, 1.0)
  ax.set_ylabel("Score")
  ax.set_title("Comparação de Modelos – Métricas com Threshold 0.5")
  ax.legend(loc='upper left')
  ax.grid(axis="y", linestyle="--", alpha=0.4)

  plt.tight_layout()
  out_graph = os.path.join("Comparações", "comparacao_modelos.png")
  plt.savefig(out_graph, dpi=300)
  print(f"Gráfico principal salvo como: {out_graph}")

  plt.close()

  # =======================
  # REMOVIDOS:
  # - GRÁFICO ADICIONAL – MÉTRICAS COM THRESHOLD OTIMIZADO
  # - GRÁFICOS POR MÉTRICA – 0.5 x THRESHOLD OTIMIZADO
  # =======================

  print("\n✅ Todos os gráficos principais foram gerados com sucesso!")

if __name__ == "__main__":
  main()

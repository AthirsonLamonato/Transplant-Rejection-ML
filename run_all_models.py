import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from base_functions import carregar_dados_com_subamostragem, tratar_features, RANDOM_STATE, CLASS_WEIGHT
from model_decision_tree import treinar_modelo as treinar_dt
from model_random_forest import treinar_modelo as treinar_rf
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

  # Remove entradas None (caso algum modelo falhe)
  metrics_list = [m for m in metrics_list if m is not None]

  df_metrics = pd.DataFrame(metrics_list)
  out_name = "comparacao_modelos.csv"
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
  }

  for k, v in explicacoes.items():
    print(f"{k:<15} => {v}")

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
  # GRÁFICO COM MATPLOTLIB
  # =======================
  print("\nGerando gráfico de comparação com matplotlib...")

  # Métricas que vamos plotar
  metrics_to_plot = ["roc_auc_mean", "recall_mean", "precision_mean", "f1_mean", "accuracy_mean"]
  metric_labels = ["ROC-AUC", "Recall", "Precision", "F1-Score", "Accuracy"]

  # Eixo X: uma posição para cada métrica
  x = np.arange(len(metrics_to_plot))
  n_models = len(df_metrics)
  width = 0.25  # largura de cada barra

  fig, ax = plt.subplots(figsize=(10, 6))

  for i, row in df_metrics.iterrows():
    model_name = row["model"]
    values = [row[m] for m in metrics_to_plot]
    ax.bar(x + i * width, values, width, label=model_name)

  ax.set_xticks(x + width * (n_models - 1) / 2)
  ax.set_xticklabels(metric_labels)
  ax.set_ylim(0.0, 1.0)
  ax.set_ylabel("Score")
  ax.set_title("Comparação de Modelos – Métricas Médias (Cross-Validation)")
  ax.legend()
  ax.grid(axis="y", linestyle="--", alpha=0.4)

  plt.tight_layout()
  plt.savefig("comparacao_modelos.png", dpi=300)
  print("Gráfico salvo como: comparacao_modelos.png")

  plt.show()

if __name__ == "__main__":
  main()
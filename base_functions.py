import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score)

# ===== CONFIGURAÇÕES GLOBAIS =====
CACHE_FILE = "cache_treino.parquet" # Arquivo de cache já pré-processado
CSV_FILE = "./base_de_dados.csv" # Caminho para o CSV original
TARGET = "REJECTION" # Coluna alvo

MAJORITY_SAMPLE_SIZE = 50000 # Máximo de amostras da classe 0 (sem rejeição)
CSV_CHUNK_SIZE = 50000 # Tamanho do chunk para ler o CSV

FEATURE_IMPORTANCE_CSV = "feature_importances.csv"
CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42

# ===== FUNÇÕES DE BASE =====

def processar_df(df: pd.DataFrame):
  """Valida o DataFrame, garante que TARGET existe e é 0/1, filtra linhas inválidas."""
  if df.empty:
      raise SystemExit("[ERRO] DataFrame vazio.")

  target_col = TARGET if TARGET in df.columns else "REJECTION"
  if target_col not in df.columns:
    raise SystemExit(f"[ERRO] DataFrame sem coluna TARGET '{TARGET}' ou 'REJECTION'.")

  y = df[target_col]
  y_num = pd.to_numeric(y, errors="coerce")
  mask_valid = y_num.isin([0, 1])

  antes = len(df)
  df = df[mask_valid].copy()
  depois = len(df)

  if depois == 0:
    return None

  removidas = antes - depois
  if removidas > 0:
    print(
      f"[INFO] {antes} {depois} linhas "
      f"(removidas {removidas} com alvo diferente de 0/1 ou nulo)."
    )

  # Garante tipo inteiro limpo na própria coluna TARGET
  df[target_col] = y_num[mask_valid].astype(int)
  return df

def carregar_dados_com_subamostragem():
  """Carrega dados do cache ou CSV, faz subamostragem da classe majoritária e retorna df."""
  if os.path.exists(CACHE_FILE):
    print(f"Carregando dados do cache: {CACHE_FILE}")
    df_full = pd.read_parquet(CACHE_FILE)

    # Compatibilidade com cache antigo que tinha ALVO
    if TARGET not in df_full.columns and "ALVO" in df_full.columns:
      print("[AVISO] Cache antigo detectado com coluna 'ALVO'. Renomeando para 'REJECTION'.")
      df_full.rename(columns={"ALVO": TARGET}, inplace=True)

    if TARGET not in df_full.columns:
      raise SystemExit(f"Cache não contém coluna alvo '{TARGET}'. Remova o cache e rode de novo.")

    mem = df_full.memory_usage(deep=True).sum() / 1024**2
    print(f"\nTotal acumulado: {df_full.shape[0]} linhas | Memória: {mem:.2f} MB")
    return df_full

  print(f"Carregando dados do CSV em chunks: {CSV_FILE}")

  minority_frames = []
  majority_frames = []
  majority_count = 0

  try:
    reader = pd.read_csv(CSV_FILE, chunksize=CSV_CHUNK_SIZE, low_memory=False)
  except Exception as e:
    raise SystemExit(f"[ERRO] Falha ao ler o CSV: {e}")

  for idx, df_chunk in enumerate(reader):
    print(f"Processando chunk {idx + 1}...")

    # 1. Limpeza / validação do alvo
    df_chunk = processar_df(df_chunk)
    if df_chunk is None:
      continue

    # 2. Separação e subamostragem
    df_minority = df_chunk[df_chunk[TARGET] == 1]
    df_majority = df_chunk[df_chunk[TARGET] == 0]

    # Guarda toda classe minoritária (1 - rejeição)
    if not df_minority.empty:
      minority_frames.append(df_minority)

    # Amostra classe majoritária (0 - sem rejeição) até o limite
    remaining_needed = MAJORITY_SAMPLE_SIZE - majority_count
    if remaining_needed > 0:
      sample_size = min(len(df_majority), remaining_needed)

      if sample_size > 0:
        df_sampled = df_majority.sample(n=sample_size, random_state=RANDOM_STATE)
        majority_frames.append(df_sampled)
        majority_count += sample_size

    if majority_count >= MAJORITY_SAMPLE_SIZE:
      print(f"Limite de {MAJORITY_SAMPLE_SIZE} amostras da classe majoritária atingido.")

  # 3. Concatenação final
  if not minority_frames and not majority_frames:
    raise SystemExit("Nenhum dado válido foi processado.")

  df_full = pd.concat(minority_frames + majority_frames, ignore_index=True)

  mem = df_full.memory_usage(deep=True).sum() / 1024**2
  print(f"\nTotal acumulado: {df_full.shape[0]} linhas | Memória: {mem:.2f} MB")

  df_full.to_parquet(CACHE_FILE, index=False)
  print(f"Dados salvos em {CACHE_FILE}")

  return df_full

def tratar_features(df: pd.DataFrame) -> pd.DataFrame:
  """
  Converte colunas Y/N/U para 0/1/NaN e textos numéricos para float/int,
  exceto colunas de antígeno BW4/BW6, que são convertidas para categorias reais
  para evitar interpretações numéricas incorretas.
  """

  # Colunas a serem tratadas
  cols = [c for c in df.columns if c != TARGET]

  # Valores que representam Y/N/U e variações
  ynu_set = {"Y", "N", "y", "n", "U", "u", "OTHER", "Null or Missing", "NULL", "MISSING"}

  # Listas para log
  conv_yn = []
  conv_num = []
  conv_bw = []

  # --- Tratamento especial para BW6 e BW4 ---

  bw_map = {
    "0": "absent",
    "95": "positive",
    "96": "negative",
    "98": "blank",
    "99": "not_tested",
    "998": "unknown",
    "OTHER": "unknown",
    "Null or Missing": "missing",
    "NULL": "missing",
    "MISSING": "missing"
  }

  for bw_col in ["BW6", "BW4"]:
    if bw_col in df.columns:
      df[bw_col] = (
        df[bw_col]
        .astype(str)
        .str.strip()
        .map(bw_map)
        .fillna("missing")
        .astype("object")
      )

    conv_bw.append(bw_col)

  # --- Demais tratamentos ---

  for c in cols:
    if c in ["BW6", "BW4"]:
      continue  # já tratadas acima

    s = df[c]

    # Se já for número, pula
    if np.issubdtype(s.dropna().dtype, np.number):
      continue

    s_nonnull = s.dropna()
    if s_nonnull.empty:
      continue

    vals = set(map(str.strip, map(str, s_nonnull.unique())))

    # Y/N/U → 0/1/NaN
    if vals.issubset(ynu_set):
      mapa = {
        "Y": 1, "y": 1,
        "N": 0, "n": 0,
        "U": np.nan, "u": np.nan,
        "OTHER": np.nan,
        "Null or Missing": np.nan,
        "NULL": np.nan,
        "MISSING": np.nan
      }
      df[c] = s.astype(str).str.strip().map(mapa)
      conv_yn.append(c)
      continue

    # Textos numéricos → numérico (quando >50% são números válidos)
    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.notna().sum() / s_nonnull.count() >= 0.5:
      df[c] = coerced
      conv_num.append(c)

  # Logs
  if conv_bw:
    print(f"\nColunas BW4/BW6 convertidas para categorias ({len(conv_bw)}): {conv_bw}")

  if conv_yn:
    print(f"\nColunas convertidas de Y/N/U/OTHER/Null or Missing para 0/1/NaN ({len(conv_yn)}):")
    print(conv_yn[:20], "..." if len(conv_yn) > 20 else "")

  if conv_num:
    print(f"\nColunas convertidas de texto para numérico ({len(conv_num)}):")
    print(conv_num[:20], "..." if len(conv_num) > 20 else "")

  return df


def get_preprocessor(X, num_cols, cat_cols):
  """
  Monta o pré-processador dos dados.

  - Numéricas: valores faltantes são preenchidos pela mediana e os dados são padronizados
    com StandardScaler, que coloca todas as variáveis na mesma escala (média 0 e desvio 1).

  - Categóricas: valores ausentes são preenchidos pelo valor mais frequente e convertidos
    em vetores binários por One-Hot Encoding, permitindo que o modelo trate categorias
    como informações independentes sem criar ordem entre elas.
  
  O ColumnTransformer coordena essas etapas, aplicando o tratamento apropriado para cada grupo de colunas.
  """

  pre = ColumnTransformer(
    transformers=[
      # ---- TRATAMENTO PARA VARIÁVEIS NUMÉRICAS ----
      (
        "num",
        Pipeline(steps=[
          ("imp", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler())
        ]),
        num_cols
      ),

      # ---- TRATAMENTO PARA VARIÁVEIS CATEGÓRICAS ----
      (
        "cat",
        Pipeline(steps=[
          ("imp", SimpleImputer(strategy="most_frequent")),
          ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]),
        cat_cols
      ),
    ],

    remainder="drop",
  )

  return pre

def get_data_and_features(df: pd.DataFrame):
  """Separa X, y, remove leakage conhecido e define listas de colunas numéricas e categóricas."""
  if TARGET not in df.columns:
    raise SystemExit(f"[ERRO] DataFrame sem coluna alvo '{TARGET}'.")

  y = df[TARGET]
  X = df.drop(columns=[TARGET])

  # === 1) Colunas de leakage que você JÁ definiu ===
  leak_cols = [
    "ACUTE_REJ_EPI_KI",
    "REJCNF_KI",
    "REJTRT_KI",
    "TRTREJ6M_KI",
    "TRTREJ1Y_KI",
    "REJ_BIOPSY",
    "GRF_FAIL_CAUSE_TY_KI",
    "LOS",
    "FIRST_WK_DIAL",
    "REJECTION_LABEL",
    "REJECTION_SOURCE",
    "CREAT6M",
    "TRR_ID",
    "TRR_ID_CODE",
    "PT_CODE",
    "SERUM_CREAT",
    "GTIME_KI",
    "GRF_STAT_KI",
    "GSTATUS_KI",
    "CREAT1Y",
    "CREAT_DON",
    "CREAT_TRR",
    "TXPAN_None",
    "TXPAN_W",
    "DIABETES_DON",
    "HIST_DIABETES_DON",
    "INSULIN_DON",
    "INSULIN_DEP_DON",
    "HIST_INSULIN_DEP_DON",
    "INSULIN_DUR_DON",
    "PRE_AVG_INSULIN_USED_OLD_TRR",
    "TX_PAN_W",
    "TX_PAN",
    "EDUCATION",
    "EDUCATION_DON",
    "DONOR_ID",
    "ORGAN_KI",
    "ORGAN_KP",
    "ORGAN"
  ]

  # === 2) Extra: garantir que QUALQUER coisa de TXPAN saia (variações de nome) ===
  # Isso é só pra cobrir o caso TXPAN_W virar algo ligeiramente diferente no CSV.
  txpan_cols = [c for c in X.columns if "TXPAN" in c.upper() or "TX_PAN" in c.upper()]

  # === 3) IDs genéricos que você já queria tirar ===
  extra_ids = []
  if "_id" in X.columns:
    extra_ids.append("_id")

  cols_to_drop = [c for c in leak_cols if c in X.columns]
  cols_to_drop = sorted(set(cols_to_drop + txpan_cols + extra_ids))

  if cols_to_drop:
    print("\n[INFO] Removendo colunas de leakage/ID do modelo:")
    print(f"Total: {len(cols_to_drop)}")
    print(cols_to_drop)
    X = X.drop(columns=cols_to_drop, errors="ignore")

  # separa tipos
  num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

  cat_cols_full = [
    c
    for c in X.columns
    if (pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]))
    and c not in num_cols
  ]

  # Filtro de cardinalidade: mantém apenas colunas com até 50 valores únicos
  cat_cols = [c for c in cat_cols_full if X[c].nunique() <= 50]

  ignored_cat_cols = [c for c in cat_cols_full if c not in cat_cols]
  if ignored_cat_cols:
    print(f"[AVISO] Ignorando {len(ignored_cat_cols)} colunas categóricas de alta cardinalidade (>50): {ignored_cat_cols[:5]}...")
    X = X.drop(columns=ignored_cat_cols, errors="ignore")

  print(f"\nNuméricas: {len(num_cols)} | Categóricas (<=50): {len(cat_cols)}")

  return X, y, num_cols, cat_cols


def evaluate_model(clf, X_test, y_test, model_name):
  pred = clf.predict(X_test)

  print("\n==================================================")
  print(f"        MÉTRICAS DE AVALIAÇÃO - {model_name}")
  print("==================================================")

  # ROC-AUC
  roc_auc = None
  try:
    proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    print("ROC-AUC:", f"{roc_auc:.4f}")
  except Exception:
    print("ROC-AUC: Não disponível (modelo não suporta predict_proba)")

  recall_rejection = recall_score(y_test, pred, pos_label=1)
  acc = accuracy_score(y_test, pred)

  print("Recall (Sensibilidade - Classe 1/Rejeição):",f"{recall_rejection:.4f}")
  print("Acurácia (Apenas Informativa):", f"{acc:.4f}")

  print("\nRelatório de classificação (Foco em Precision, Recall, F1-Score):\n")

  report = classification_report(
    y_test,
    pred,
    digits=4,
    target_names=["Sem Rejeição (0)", "Com Rejeição (1)"],
  )
  print(report)

  cm = confusion_matrix(y_test, pred)
  print("Matriz de confusão (FN é o mais crítico):\n", cm)

  tn, fp, fn, tp = cm.ravel()

  metrics_dict = {
    "model": model_name,
    "roc_auc": float(roc_auc) if roc_auc is not None else None,
    "recall": float(recall_rejection),
    "accuracy": float(acc),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
  }

  # Importância de features
  try:
    if hasattr(clf.named_steps["model"], "feature_importances_"):
      model_step = clf.named_steps["model"]
      importances = model_step.feature_importances_
      preprocessor = clf.named_steps["prep"]

      try:
        raw_feature_names = list(preprocessor.get_feature_names_out())
      except Exception:
        raw_feature_names = [f"feat_{i}" for i in range(len(importances))]

      if len(raw_feature_names) != len(importances):
        raw_feature_names = [f"feat_{i}" for i in range(len(importances))]

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

      imp_df = (pd.DataFrame({"feature": clean_names, "feature_type": feature_type, "importance": importances}).sort_values("importance", ascending=False))

      imp_df["importance"] = imp_df["importance"] * 100

      top_n = min(30, len(imp_df))
      print(f"\nTop {top_n} variáveis mais importantes (em %):")
      print(
        imp_df.head(top_n)
        .rename(columns={"importance": "importance_pct"})
        .to_string(index=False)
      )

      imp_df.rename(columns={"importance": "importance_pct"}, inplace=True)
      imp_df.to_csv(
        f"feature_importances_{model_name.lower().replace(' ', '_')}.csv",
        index=False,
      )
      print(f"\nImportâncias salvas em feature_importances_{model_name.lower().replace(' ', '_')}.csv")
  except Exception as e:
    print(f"[AVISO] Não foi possível calcular importâncias de features para {model_name}: {e}")

  return metrics_dict


def main_template(treinar_modelo_func, model_name):
  print(f"Iniciando Treinamento: {model_name}")
  print("Lendo e processando dados...")
  df = carregar_dados_com_subamostragem()

  print(f"\nDistribuição do alvo ({TARGET}):")
  print(df[TARGET].value_counts())
  print("\nDistribuição percentual:")
  print(df[TARGET].value_counts(normalize=True) * 100)

  print("\nTratando features (Y/N/U/OTHER, textos numéricos, etc.)...")
  df = tratar_features(df)

  print(f"\nTreinando modelo {model_name}...")
  metrics = treinar_modelo_func(df, model_name)

  print(f"\n{model_name} Finalizado com sucesso.")
  return metrics

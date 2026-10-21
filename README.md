# Machine Learning para Predição Pré-Transplante de Rejeição Renal

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/AthirsonLamonato/Transplant-Rejection-ML/blob/main/MIT%20License)
[![Status](https://img.shields.io/badge/Status-Concluído-success.svg)]()

> Trabalho de Conclusão de Curso - Instituto Federal de Educação, Ciência e Tecnologia do Rio Grande do Sul

## 📋 Sobre o Projeto

Este projeto desenvolve e avalia modelos preditivos de rejeição do enxerto renal utilizando técnicas de Aprendizado de Máquina aplicadas exclusivamente a dados pré-transplante. O objetivo é auxiliar equipes médicas na estratificação de risco e tomada de decisões clínicas.

### 🎯 Principais Resultados

- **Melhor Modelo:** XGBoost
- **AUC-ROC:** 0.890
- **Sensibilidade:** 82.6% (após otimização de threshold)
- **Base de Dados:** OPTN/UNOS STAR (64.764 transplantes)
- **Variáveis:** 342 features pré-transplante

## 🔬 Metodologia

### Base de Dados

- **Fonte:** National STAR File (OPTN/UNOS)
- **Período:** 1987-2024
- **Registros Totais:** ~1.2M
- **Registros Utilizados:** 64.764 (após limpeza e balanceamento)
- **Classes:**
  - Sem Rejeição: 50.000
  - Com Rejeição: 14.764

### Pipeline de Processamento

```
Dados Brutos (1.2M registros)
    ↓
Integração de Tabelas (HLA, PRA, Dados Clínicos)
    ↓
Filtragem Pré-Transplante (342 variáveis)
    ↓
Prevenção de Data Leakage
    ↓
Balanceamento (Undersampling)
    ↓
Pipeline de Pré-processamento Automático
    ↓
Validação Cruzada Estratificada (5-folds)
    ↓
Otimização de Threshold (F2-Score)
```

## 🤖 Modelos Avaliados

| Modelo              | AUC-ROC         | Acurácia | Precisão | Recall | F1-Score        |
| ------------------- | --------------- | --------- | --------- | ------ | --------------- |
| **XGBoost**   | **0.890** | 0.873     | 0.736     | 0.687  | **0.711** |
| Random Forest       | 0.869           | 0.882     | 0.867     | 0.570  | 0.687           |
| RNA (MLP)           | 0.867           | 0.886     | 0.938     | 0.539  | 0.683           |
| Árvore de Decisão | 0.844           | 0.809     | 0.562     | 0.691  | 0.620           |

### Desempenho após Otimização de Threshold

| Modelo              | Threshold      | Precisão       | Recall          | F2-Score        | Acurácia       |
| ------------------- | -------------- | --------------- | --------------- | --------------- | --------------- |
| **XGBoost**   | **0.35** | **0.504** | **0.826** | **0.732** | **0.775** |
| Random Forest       | 0.38           | 0.465           | 0.809           | 0.705           | 0.744           |
| RNA (MLP)           | 0.15           | 0.454           | 0.811           | 0.701           | 0.735           |
| Árvore de Decisão | 0.32           | 0.405           | 0.819           | 0.680           | 0.684           |

## 📊 Variáveis Mais Importantes

### Top 10 Features (XGBoost)

1. **BW6_absent** (23.2%) - Marcador HLA
2. **BW4_absent** (22.8%) - Marcador HLA
3. **CMV_IGG_N** (2.7%) - Sorologia Citomegalovírus
4. **FUNC_STAT_TRR** (2.3%) - Estado Funcional do Receptor
5. **HBSAB_DON_None** (2.1%) - Hepatite B do Doador
6. **HBV_CORE_DON_N** (1.9%) - Marcador Hepatite B
7. **USE_WHICH_PRA_C** (1.7%) - Tipo de Painel de Anticorpos
8. **EBV_SEROSTATUS_None** (1.6%) - Epstein-Barr
9. **TOT_SERUM_ALBUM** (1.5%) - Albumina Sérica
10. **BW6_positive** (1.4%) - Marcador HLA

### Categorias de Variáveis

- **Compatibilidade Imunológica:** HLA, anticorpos, crossmatch
- **Características do Receptor:** idade, peso, tempo em diálise, comorbidades
- **Características do Doador:** tipo (vivo/falecido), idade, causa da morte
- **Fatores Procedimentais:** ano, região, tipo de cirurgia

## 🚀 Instalação e Uso

### Pré-requisitos

```bash
Python 3.10+
pip ou conda
```

### Instalação

```bash
# Clone o repositório
git clone https://github.com/AthirsonLamonato/Transplant-Rejection-ML.git
cd Transplant-Rejection-ML

# Instale as dependências
pip install -r requirements.txt
```

### Estrutura do Projeto

```
Transplant-Rejection-ML/
├── base_de_dados.csv              # Dataset processado
├── base_functions.py              # Funções auxiliares
├── model_ann.py                   # Modelo Rede Neural
├── model_decision_tree.py         # Modelo Árvore de Decisão
├── model_random_forest.py         # Modelo Random Forest
├── model_xgboost.py               # Modelo XGBoost
├── run_all_models.py              # Script para executar todos os modelos
├── Comparações/                   # Análises comparativas
│   ├── comparacao_modelos.csv
│   └── comparacao_modelos.png
├── Gráficos/                      # Visualizações
│   ├── feature_importances_*.png
│   └── heatmap_modelos_metricas.png
├── Features/                      # Feature importance por modelo
├── Resumos/                       # Resumos de treinamento
└── requirements.txt               # Dependências
```

### Executando os Modelos

```bash
# Executar todos os modelos
python run_all_models.py

# Executar modelo específico
python model_xgboost.py

# Executar com validação cruzada completa
python model_xgboost.py --cv-folds 5 --optimize-threshold
```

## 📈 Visualizações

O projeto gera automaticamente:

- Curvas ROC para todos os modelos
- Matrizes de confusão
- Gráficos de importância de features
- Mapas de calor comparativos
- Análise de threshold optimization

## 🔍 Principais Contribuições

1. **Pipeline Robusto:** Implementação completa do processo KDD com prevenção de data leakage
2. **Otimização Clínica:** Ajuste de threshold priorizando sensibilidade (F2-Score)
3. **Análise de Importância:** Identificação de marcadores HLA como principais preditores
4. **Comparação Sistemática:** Avaliação de 4 algoritmos com métricas múltiplas
5. **Código Aberto:** Todo o código disponível para reprodução e extensão

## 📚 Referências

- **Base de Dados:** OPTN/UNOS STAR File ([UNOS](https://unos.org/data/))
- **Trabalhos Correlatos:**
  - Mark et al. (2019) - Random Survival Forests
  - Kawakita et al. (2020) - Predição de DGF
  - Naqvi et al. (2021) - Sobrevivência do Enxerto

## 🎓 Autor

**Athirson Lamonato Ferreira**

- Instituto Federal do Rio Grande do Sul - Campus Ibirubá
- Orientador: Prof. Andrws Aires Vieira
- Email: [athirson.lamonato@gmail.com]
- [[LinkedIn](https://www.linkedin.com/in/athirsonlamonato/)]

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/AthirsonLamonato/Transplant-Rejection-ML/blob/main/MIT%20License) para mais detalhes.

## 🙏 Agradecimentos

- Instituto Federal do Rio Grande do Sul - Campus Ibirubá
- OPTN/UNOS pela disponibilização dos dados
- Comunidade científica de Machine Learning aplicado à saúde

## 📖 Citação

Se você utilizar este trabalho, por favor cite:

```bibtex
@mastersthesis{lamonato2025ml,
  title={Machine Learning para Predição Pré-Transplante de Rejeição Renal},
  author={Lamonato Ferreira, Athirson},
  year={2025},
  school={Instituto Federal de Educação, Ciência e Tecnologia do Rio Grande do Sul},
  type={Trabalho de Conclusão de Curso}
}
```

---

**Nota:** Este é um projeto acadêmico desenvolvido para fins de pesquisa. Os modelos não devem ser utilizados para decisões clínicas reais sem validação adicional e aprovação regulatória apropriada.

## 🔮 Trabalhos Futuros

- [ ] Validação em bases de dados brasileiras
- [ ] Inclusão de dados pós-transplante
- [ ] Desenvolvimento de interface web
- [ ] Modelos especializados por subgrupos
- [ ] Avaliação com equipes médicas
- [ ] Análises temporais (séries temporais)

## 📞 Contato

Para dúvidas, sugestões ou colaborações:

- **Issues:** [GitHub Issues](https://github.com/AthirsonLamonato/Transplant-Rejection-ML/issues)
- **Email:** athirson.lamonato@gmail.com

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!

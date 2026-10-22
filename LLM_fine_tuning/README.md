# 🧩 LLM LoRA Fine-tuning — Text-to-SQL sur CPU

Fine-tuning LoRA/QLoRA d'un LLM léger (Qwen2.5-1.5B-Instruct) pour générer
une requête SQL à partir d'une question en langage naturel et d'un schéma
`CREATE TABLE`, puis export quantifié (GGUF) pour une inférence locale sur
CPU, sans GPU dédié (compatible AMD).

Dataset réel : [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)
(téléchargé automatiquement, sous-échantillon reproductible).

## Compétences démontrées

Préparation de données · fine-tuning PEFT (LoRA/QLoRA) · quantization 4-bit ·
export GGUF · inférence optimisée CPU · évaluation quantitative (exact match,
validité SQL via SQLite) · démo Streamlit · contrainte de ressources réaliste.

## Stack

`Python` · `Transformers` · `PEFT` · `bitsandbytes` · `TRL` · `datasets` ·
`llama.cpp` / `llama-cpp-python` · `SQLite` · `Streamlit`

## ⚠️ Répartition des ressources (important pour AMD)

| Étape | Script / Notebook | Ressource |
|---|---|---|
| Téléchargement + préparation du dataset | `src/download_dataset.py`, `01_data_exploration.ipynb` | 🖥️ Local CPU |
| Entraînement LoRA | `src/train.py`, `02_training_colab.ipynb` | ☁️ **Colab / Kaggle GPU (T4) — obligatoire** |
| Fusion adapters | `src/merge_and_export.py` | ☁️ Colab (recommandé) ou 🖥️ local si RAM suffisante |
| Conversion GGUF + quantization | `llama.cpp` | 🖥️ Local CPU |
| Inférence / évaluation / démo | `src/inference.py`, `src/evaluate.py`, `app/demo_streamlit.py` | 🖥️ Local CPU (AMD OK) |

QLoRA (`bitsandbytes`) n'est pas fiable sur AMD en local : l'entraînement est
la seule étape qui exige un GPU NVIDIA (Colab/Kaggle gratuits).

## Structure

```
llm-lora-finetuning/
├── data/processed/                # train.jsonl / val.jsonl (générés, ignorés par git)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # 🖥️ local
│   ├── 02_training_colab.ipynb    # ☁️ Colab GPU
│   └── 03_evaluation.ipynb        # 🖥️ local
├── src/
│   ├── prompts.py                 # format de prompt partagé train/inférence
│   ├── download_dataset.py        # téléchargement HF + split
│   ├── train.py                   # entraînement LoRA (Colab)
│   ├── merge_and_export.py        # fusion adapters
│   ├── inference.py               # inférence CLI (GGUF)
│   └── evaluate.py                # métriques
├── app/demo_streamlit.py
├── docs/                          # résultats JSON, rapport
├── models/                        # poids (ignorés par git)
├── requirements-local.txt
├── requirements-colab.txt
└── .gitignore
```

## Pipeline

### 1. Installation locale
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-local.txt
```

### 2. Dataset (local)
```bash
python src/download_dataset.py --n_train 2000 --n_val 200 --out data/processed
```

### 3. Entraînement (Colab GPU)
Pousser le repo sur GitHub, ouvrir `notebooks/02_training_colab.ipynb` dans
[Colab](https://colab.research.google.com/) (runtime **GPU T4**), remplacer
`<TON_USERNAME>` par ton identifiant GitHub, uploader `train.jsonl`/`val.jsonl`
et exécuter. Télécharger ensuite les adapters (quelques Mo).

### 4. Fusion + conversion GGUF
```bash
python src/merge_and_export.py --output_dir models/merged_model   # Colab recommandé
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt
python convert_hf_to_gguf.py ../models/merged_model --outfile ../models/model-f16.gguf
./llama-quantize ../models/model-f16.gguf ../models/model-q4_k_m.gguf Q4_K_M
```
Le binaire `llama-quantize` s'obtient en compilant llama.cpp (cmake) ou via
les releases ; pour le baseline, répéter la conversion sur le modèle de base
seul (sans adapters) → `models/base-model-q4_k_m.gguf`.

### 5. Inférence, évaluation, démo (CPU)
```bash
python src/inference.py --model models/model-q4_k_m.gguf \
  --context "CREATE TABLE employes (nom VARCHAR, salaire INTEGER, poste VARCHAR)" \
  --question "Quel est le salaire moyen des Data Scientists ?"

python src/evaluate.py --model models/model-q4_k_m.gguf --out docs/results_finetuned.json
python src/evaluate.py --model models/base-model-q4_k_m.gguf --out docs/results_base.json

streamlit run app/demo_streamlit.py
```

## Métriques

- **Exact match** : requête identique à la référence après normalisation
  (casse, espaces, `;` final). Stricte : une requête équivalente mais écrite
  autrement est comptée fausse.
- **SQL valide** : la requête passe `EXPLAIN` sur une base SQLite en mémoire
  construite depuis le schéma (syntaxe + tables/colonnes existantes). Ne prouve
  pas que le résultat est correct.

## Résultats

*À compléter avec les chiffres réels après entraînement et évaluation.*

| Modèle | Exact match | SQL valide |
|---|---|---|
| Base (sans fine-tuning) | — | — |
| Fine-tuné (LoRA, 2 epochs, 2000 exemples) | — | — |

## Limites et pistes

- Le dataset ne contient pas de données : pas d'évaluation par exécution
  (execution accuracy) ; l'ajouter avec Spider/BIRD serait plus rigoureux.
- Exact match sensible à l'écriture : prévoir une comparaison par résultats d'exécution.
- Comparer plusieurs rangs LoRA (r=8/16/32) ou tailles de modèle (0.5B/1.5B/3B).

## Auteur

Lyheng — [github.com/LyhengLAO](https://github.com/LyhengLAO)

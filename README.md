# Clinical Lab Report Analyser

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: HuggingFace & PEFT](https://img.shields.io/badge/HuggingFace-PEFT%20%2F%20QLoRA-orange.svg)](https://huggingface.co/)
[![Vector Store: ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Institution: Metropolia UAS](https://img.shields.io/badge/Institution-Metropolia%20UAS-black.svg)](https://www.metropolia.fi/)

An end-to-end clinical natural language processing and diagnostic analysis pipeline designed to translate unstructured electronic health records into grounded, structured laboratory summaries. The system combines **Parameter-Efficient Fine-Tuning (PEFT/QLoRA)** on open-source language models with a **Retrieval-Augmented Generation (RAG)** vector layer to eliminate hallucinations around demographic-specific numerical reference ranges.

Developed as a capstone project for the Data Handling and Machine Learning curriculum at Metropolia University of Applied Sciences.

---

## Architecture Overview

```text
┌─────────────────────────────────┐
│   Raw Clinical Observations     │  <-- Synthea EHR logs (observations.csv)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐      ┌─────────────────────────────┐
│  Data Ingestion & Curation      │ ---> │ Automated Quality Assurance │
│  (src/dataset_builder.py)       │      │ (tests/range_checker_test)  │
└────────────────┬────────────────┘      └─────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Structured JSONL Datasets    │  <-- train.jsonl, val.jsonl, test.jsonl
└────────┬────────────────┬───────┘
         │                │
         ▼                ▼
┌─────────────────┐  ┌─────────────────────────────────────────────────┐
│ QLoRA Fine-Tune │  │ ChromaDB Vector Index                           │
│ (Qwen Base)     │  │ (Demographic intervals, age/sex reference norms)│
└────────┬────────┘  └────────────────┬────────────────────────────────┘
         │                            │
         └─────────────┬──────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Grounded Diagnostic Inference Engine         │
│ - Exact boundary validation                  │
│ - Zero-shot hallucination prevention         │
│ - Clinically consistent recommendations      │
└──────────────────────────────────────────────┘
```

---

## Technical Highlights

* **Automated Data Validation:** Pre-training pipeline enforces strict unit and regression testing (`pytest`) to prevent data leakage, filter duplicate records, and check boundary integrity across demographic splits.
* **Efficient Adaptation via PEFT/QLoRA:** Fine-tunes low-rank adapters (LoRA) on top of quantized base language models (Qwen) using Google Colab GPU runtimes, preserving computational efficiency without sacrificing output structure.
* **Dual-Layer Factual Grounding (RAG):** Decouples general reasoning from static numerical recall by indexing clinical reference cutoffs in ChromaDB, retrieving dynamic patient-specific baselines at inference time.
* **Reproducible Pipeline:** Modular directory design separating raw data processing, test suites, training notebooks, and inference queries.

---

## Project Structure

```text
├── data/
│   ├── raw/                       # Raw Synthea observations dataset
│   └── processed/                 # Curated instruction-tuning sets
│       ├── train.jsonl            # Training set (80%)
│       ├── val.jsonl              # Validation set (10%)
│       └── test.jsonl             # Unseen evaluation set (10%)
├── notebooks/
│   └── model_train_lora.ipynb     # QLoRA fine-tuning workflow on Google Colab
├── RAG/
│   ├── chromadb_indexer.py        # Embeds clinical reference intervals into ChromaDB
│   └── query_engine.py            # Retrieval-augmented query execution script
├── src/
│   ├── synthea_inspector.py       # EDA, distribution profiling, and data summary
│   ├── range_checker.py           # Core logic for biomarker range classification
│   └── dataset_builder.py         # Transforms CSV records into JSONL prompt pairs
├── tests/
│   ├── duplicate_check.py         # Validates sample deduplication
│   ├── gender_test.py             # Validates sex-stratified reference thresholds
│   ├── hb_dist_test.py            # Validates hemoglobin distribution profiles
│   └── range_checker_test.py      # Assertions on biomarker status logic
├── requirements.txt               # Pinned project dependencies
└── README.md                      # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/SaileshK1103/data_handling_project_26.git](https://github.com/SaileshK1103/data_handling_project_26.git)
cd data_handling_project_26
```

### 2. Environment Configuration
Create an isolated Python 3.10+ virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage Guide

### Phase 1: Data Curation & Quality Verification
1. Inspect the source clinical data distributions:
   ```bash
   python src/synthea_inspector.py
   ```
2. Build the instruction-tuning splits:
   ```bash
   python src/dataset_builder.py
   ```
3. Run the automated data integrity and range check tests:
   ```bash
   pytest tests/
   ```

### Phase 2: Parameter-Efficient Fine-Tuning
1. Launch `notebooks/model_train_lora.ipynb` in Google Colab or on a local CUDA-enabled environment.
2. Verify hardware acceleration (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Load the pre-processed splits from `data/processed/`.
4. Train the LoRA adapter checkpoints and save the model weights.

### Phase 3: Semantic Indexing & RAG Inference
1. Index clinical biomarker ranges and reference documents into ChromaDB:
   ```bash
   python RAG/chromadb_indexer.py
   ```
2. Execute a diagnostic interpretation query:
   ```bash
   python RAG/query_engine.py --patient_age 48 --patient_sex "M" --biomarkers "Hemoglobin: 11.4 g/dL, Fasting Blood Glucose: 138 mg/dL"
   ```

---

## Evaluation & Results

The system was evaluated against two primary criteria: **structural consistency** (correct diagnostic sections and terminology) and **factual threshold accuracy** (absence of hallucinated reference intervals).

| Pipeline Stage | Structural Validity | Reference Range Accuracy | Hallucination Rate |
| :--- | :---: | :---: | :---: |
| **Base Model (Zero-Shot)** | 62% | 58% | High |
| **Fine-Tuned (PEFT / QLoRA)** | 94% | 79% | Moderate (edge-case drift) |
| **Fine-Tuned + RAG (ChromaDB)** | **99%** | **98%** | **Near-Zero (< 2%)** |

By integrating dynamic retrieval over static memorization, the model eliminates numerical boundary errors while retaining clinical reporting fluency.

---

## Contributors

* **Sailesh Karki** – [GitHub](https://github.com/SaileshK1103) | [LinkedIn](https://www.linkedin.com/in/sailesh-karki)
* Built collaboratively as part of the **Data Handling and Machine Learning** team project at **Metropolia University of Applied Sciences**. Upstream project repository: [berraEgcin/data_handling_project_26](https://github.com/berraEgcin/data_handling_project_26).

---

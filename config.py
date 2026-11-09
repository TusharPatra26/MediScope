"""Central configuration for MediScope.

Two kinds of settings live here, deliberately kept apart:

* ``TRAIN_*``  - research/training defaults used by the notebooks. They are sized
  for a CPU-only student laptop and are meant to be overridden (for example on a
  Colab GPU) by editing the notebook's config cell.
* ``INFER_*``  - deployment settings used by the Flask app at inference time.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
ARTIFACTS = BASE_DIR / "artifacts"
ART_CLS = ARTIFACTS / "classification"
ART_NER = ARTIFACTS / "ner"
ART_SUM = ARTIFACTS / "summarization"
ART_EVAL = ARTIFACTS / "evaluation"
INSTANCE_DIR = BASE_DIR / "instance"
HISTORY_DB = INSTANCE_DIR / "history.db"

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# Datasets (see data/README.md for provenance and licensing notes)
# ----------------------------------------------------------------------------
PUBMED_RCT_BASE = (
    "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT"
)
PUBMED_RCT_FILES = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}

BC5CDR_BASE = "https://huggingface.co/datasets/tner/bc5cdr/resolve/main/dataset"
BC5CDR_FILES = {"train": "train.json", "dev": "valid.json", "test": "test.json"}
BC5CDR_LABELS = {"O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4}

SUMM_TEST_URL = (
    "https://huggingface.co/datasets/ccdv/pubmed-summarization/resolve/main/"
    "document/test-00000-of-00001.parquet"
)

# ----------------------------------------------------------------------------
# Task definitions
# ----------------------------------------------------------------------------
CLASS_LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
CLASS_DESCRIPTIONS = {
    "BACKGROUND": "Context and motivation for the study.",
    "OBJECTIVE": "The question, aim or hypothesis being tested.",
    "METHODS": "Design, participants, interventions and measurements.",
    "RESULTS": "Reported findings and statistics.",
    "CONCLUSIONS": "Authors' interpretation and implications.",
}
NER_TAGS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
NER_ENTITY_TYPES = ["Chemical", "Disease"]

# ----------------------------------------------------------------------------
# Model identifiers (pretrained weights are fetched from the Hugging Face Hub
# by the training notebooks, never at request time)
# ----------------------------------------------------------------------------
HF_DISTILBERT = "distilbert-base-uncased"
HF_BERT = "bert-base-uncased"
HF_BIOBERT = "dmis-lab/biobert-base-cased-v1.1"
# The BioBERT hub repo ships no tokenizer files; BioBERT v1.1 reuses the original cased
# BERT WordPiece vocabulary (Lee et al., 2020), so the tokenizer comes from bert-base-cased.
HF_BIOBERT_TOKENIZER = "bert-base-cased"
HF_SUMMARIZER = "Falconsai/medical_summarization"  # T5-small, Apache-2.0

# ----------------------------------------------------------------------------
# Research / training defaults (notebooks)
# ----------------------------------------------------------------------------
TRAIN_MAX_SEQ_LEN = 64            # sentences in PubMed RCT are short
TRAIN_SUBSET_SENTENCES = 3_000    # stratified training subset for transformers (CPU: several s per step)
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 1
TRAIN_LR = 5e-5
TEST_SUBSET_SENTENCES = 2_000     # shared test subset so every model is compared fairly

# ----------------------------------------------------------------------------
# Inference / deployment (Flask app)
# ----------------------------------------------------------------------------
INFER_MAX_SEQ_LEN = 64
NER_MAX_SEQ_LEN = 128
SUM_MAX_INPUT_TOKENS = 512        # T5 encoder window per chunk
SUM_CHUNK_OVERLAP_SENTENCES = 0
SUM_DEFAULT_LENGTH = "medium"
SUM_LENGTHS = {                   # generation token budgets per chunk
    "short": (24, 64),
    "medium": (48, 128),
    "long": (80, 200),
}

# Uploads and text limits
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_TEXT_CHARS = 60_000
MIN_TEXT_CHARS = 40

SECRET_KEY = os.environ.get("MEDISCOPE_SECRET_KEY", "dev-only-change-me")

# Application limits (keep request time reasonable on a CPU laptop)
MAX_SENTENCES_CLASSIFY = 250
NER_MAX_CHARS = 20_000
SUM_MAX_CHUNKS = 6
EXPLAIN_MAX_SENTENCES = 8         # occlusion is costly for neural models; linear models explain every sentence
HISTORY_MAX_ROWS = 200

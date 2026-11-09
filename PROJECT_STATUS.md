# MediScope — Project Status

_Last updated: Phase 1 (research & planning) complete. Everything below is updated at the end of each phase._

## 1. Phase tracker

| Phase | Scope | Status |
|---|---|---|
| 1 | Research, dataset/licence validation, model selection, architecture, risks | **Done** |
| 2 | Data acquisition + notebooks 01-03 | Pending |
| 3 | Shared preprocessing, classical ML, BiLSTM | Pending |
| 4 | Transformer classifier, NER, summarization | Pending |
| 5 | Notebooks 06-10, benchmarking, deployment model selection | Pending |
| 6 | Flask backend, uploads, history, exports | Pending |
| 7 | Frontend / design system | Pending |
| 8 | Final verification, docs | Pending |

## 2. Datasets (verified 2026-11-09 by downloading and counting)

| Dataset | Task | Source | Verified counts | Licence status |
|---|---|---|---|---|
| **PubMed 20k RCT** (Dernoncourt & Lee, IJCNLP 2017) | Sentence-level rhetorical-role classification (BACKGROUND / OBJECTIVE / METHODS / RESULTS / CONCLUSIONS) | `github.com/Franck-Dernoncourt/pubmed-rct` (`PubMed_20k_RCT/`) | train 180,040 sentences / 15,000 abstracts; dev 30,212 / 2,500; test 30,135 / 2,500 | **No explicit licence.** Built from MEDLINE/PubMed 2016 baseline; the repo README states some abstracts may be protected by copyright and users must satisfy copyright terms. Used for research only; **never redistributed or committed**. |
| **BC5CDR** (BioCreative V CDR, Li et al. 2016), T-NER sentence-split version | Chemical / Disease NER (BIO) | `huggingface.co/datasets/tner/bc5cdr` | train 5,228; dev 5,330; test 5,865 sentences | Hugging Face card lists `license: other`. Underlying corpus is from NCBI/BioCreative for research use. Not redistributed. |
| **PubMed summarization** (Cohan et al. 2018) via `ccdv/pubmed-summarization` (`document` config, **test split only**, 58 MB) | Evaluate long-document summarization (article -> abstract) | `huggingface.co/datasets/ccdv/pubmed-summarization` | test 6,658 articles; median article ≈ 2,596 words, median abstract ≈ 213 words | Card declares **no licence**. Derived from PubMed Central Open Access. Used for evaluation only; not redistributed. The 2.2 GB train split is deliberately **not** downloaded. |

**Design decision — what "research category" means.** PubMed 20k RCT labels the *role of each sentence inside an abstract*, not the topic of the whole abstract. MediScope therefore classifies **each sentence of a submitted abstract** into a rhetorical role and aggregates this into a per-abstract structure view. I am not relabelling it as topic classification, because that would misrepresent the data.

**Leakage:** official splits are made per abstract, so no abstract spans two splits. Notebook 01 verifies this empirically. Hyper-parameter search uses `GroupKFold` on `abstract_id` so sentences of one abstract never straddle CV folds.

## 3. Models

| Task | Models | Notes |
|---|---|---|
| Classification (classical) | MultinomialNB, Logistic Regression, Linear SVM, Random Forest, XGBoost on TF-IDF (word 1-2gram) | full 180k training sentences, CPU |
| Classification (deep) | BiLSTM (PyTorch, learned embeddings; Word2Vec-initialised variant as an ablation) | CPU |
| Classification (transformer) | DistilBERT (`distilbert-base-uncased`, Apache-2.0), BERT (`bert-base-uncased`, Apache-2.0), BioBERT (`dmis-lab/biobert-base-cased-v1.1`) | fine-tuned on a stratified 20k-sentence subset on CPU (see below) |
| NER | Dictionary/rule baseline built from training-set mentions; BioBERT token classifier | BIO alignment to first subword, other subwords masked with -100 |
| Summarization | Extractive TF-IDF/TextRank baseline; `Falconsai/medical_summarization` (T5-small, Apache-2.0, **pretrained only — not fine-tuned by me**) | evaluated with ROUGE (+ BERTScore if feasible) on a test subset |

**Licence caveat — BioBERT.** The Hugging Face card carries no licence field and the GitHub API does not auto-detect one for `dmis-lab/biobert`. I will cite the original paper (Lee et al., 2020) and flag this in the README rather than assert a licence.

## 4. Hardware reality check (measured)

* CPU: Intel i5-1135G7 (4C/8T), RAM ≈ 7.8 GB.
* GPU: NVIDIA MX450, **2 GB VRAM**; installed PyTorch is the **CPU-only** build (`torch 2.15.0+cpu`, `cuda.is_available() == False`).
* Conclusion: BERT-base full fine-tuning on the whole dataset is not realistic locally. Compromises (documented, not hidden):
  1. Transformers are fine-tuned on a **configurable stratified subset** (`TRAIN_SUBSET_SENTENCES`, default 8,000) with `max_seq_len=64`.
  2. Transformers are evaluated on a **fixed stratified 5,000-sentence test subset**; classical/BiLSTM models are reported on both the full test set and the same subset so the comparison is like-for-like.
  3. All notebook configs are single cells at the top; on Colab GPU set a larger subset and rerun.
* I did **not** install a CUDA build of PyTorch (≈2.5 GB download, and 2 GB VRAM would still be too small for BERT-base AdamW).

## 5. Architecture (final, compact)

Follows the mandated layout. `src/` holds reusable logic shared by notebooks and the app; notebooks hold experiments; the Flask app is a single `app.py` with a small number of routes (blueprints only if it grows). Preprocessing is **one function used for both training and inference** (`src/preprocessing.py`).

Additions to the suggested tree, each justified:
* `static/js/vendor/` — Chart.js (MIT) vendored locally so the app is fully offline.
* `data/README.md` — dataset provenance.
* No Docker, Redis, Celery, extra DBs.

## 6. Technical risks and mitigations

| Risk | Mitigation |
|---|---|
| CPU-only training time | Subsets, short sequence length, dynamic padding, background runs, checkpointing |
| 8 GB RAM | Sparse TF-IDF, capped features, no 180k x dense matrices; RF with limited trees/depth |
| Transformer models loaded per request | Lazy singleton loader with a lock, loaded once per process |
| Missing artifacts | Pages show exactly which notebook/command produces the artifact; **no mock predictions anywhere** |
| Uncalibrated scores | UI labels softmax/SVM scores as uncalibrated; notebook 09 measures ECE |
| Abstractive summaries can hallucinate | Extractive key sentences shown alongside; UI warning; notebook 08 includes a factual-consistency check (numeric/entity overlap) |
| PDF extraction quality | pypdf text extraction, empty/scan-only PDFs rejected with a clear message |
| Uploaded documents are sensitive | Processed in memory/temp file, deleted in `finally`; history stores only a short excerpt + results, not the file |

## 7. Files created so far

`config.py`, `src/data.py`, `PROJECT_STATUS.md`, `static/fonts/*.woff2` (OFL-1.1), `static/js/vendor/chart.umd.min.js` (MIT).

## 8. Model training status

None trained yet.

## 9. Dataset preparation status

Raw data downloaded to `data/raw/` (gitignored) via `python -m src.data`. Counts verified (section 2).

## 10. Test results

No tests yet.

## 11. Known issues

* Torch is CPU-only; see section 4.

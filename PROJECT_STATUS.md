# MediScope — Project Status

_All phases complete. Numbers below come from the executed notebooks; nothing is estimated._

## 1. Phase tracker

| Phase | Scope | Status |
|---|---|---|
| 1 | Research, dataset/licence validation, model selection, architecture, risks | Done |
| 2 | Data acquisition + notebooks 01–03 | Done (executed) |
| 3 | Shared preprocessing, classical ML, BiLSTM | Done (notebooks 04–05 executed) |
| 4 | Transformer classifier, NER, summarization | Done (notebooks 06–08 executed) |
| 5 | Explainability, benchmarking, deployment selection | Done (notebooks 09–10 executed) |
| 6 | Flask backend, uploads, history, exports | Done |
| 7 | Frontend / design system (clinical teal–crimson palette, sans + serif typography, footer) | Done |
| 8 | Final verification, docs | Done — see test results |

## 2. Datasets (verified by downloading and counting)

| Dataset | Verified counts | Licence status |
|---|---|---|
| PubMed 20k RCT | train 180,040 sentences / 15,000 abstracts; dev 30,212 / 2,500; test 30,135 / 2,500 | no explicit licence; MEDLINE-derived, some abstracts may be copyrighted; research use only |
| BC5CDR (T-NER) | 5,228 / 5,330 / 5,865 sentences | hub card: "other" |
| PubMed summarization (test split only) | 6,658 articles | none declared |

Not committed to Git; `python -m src.data` re-creates them. Full provenance: `data/README.md`. "Research category" = rhetorical role of each sentence (what PubMed RCT labels).

## 3. Model training status (CPU-only machine; budgets are small and stated)

| Task | Model | Budget | Result |
|---|---|---|---|
| Classification | NB, LR, Linear SVM | all 180,040 sentences | macro-F1 0.743 / 0.771 / 0.789 |
| Classification | RF, XGBoost, LR reference | 30,000 sentences | 0.694 / 0.690 / 0.747 |
| Classification | BiLSTM | 30,000 sentences | 0.762 |
| Classification | DistilBERT / BERT-base / **BioBERT** | 3,001 sentences, 1 epoch | 0.743 / 0.748 / **0.799** (deployed) |
| NER | BioBERT token classifier | 1,000 sentences, 2 epochs | entity F1 0.850 (dictionary on same data 0.595; on all data 0.775) |
| Summarization | `Falconsai/medical_summarization` (T5-small) | **pretrained only, not fine-tuned** | ROUGE-1 0.308 on 15 articles |

All classification scores: shared 2,000-sentence test subset. Deployment rule and its post-hoc revision are documented in notebook 10.

## 4. Deviations from the original brief (explained, not hidden)

* **Compute:** no CUDA PyTorch and a 2 GB GPU → transformers/NER trained on small subsets (configurable in `config.py`/notebook config cells); summarizer not fine-tuned.
* **Hyper-parameter studies** for transformers are one-factor-at-a-time on a 600-sentence budget; epoch/large-batch sweeps were not run.
* **Seeds:** single seed per experiment (3 → 2 seeds in notebook 03 for time); small differences are within noise.
* **Latency** was measured while other notebooks ran on the same CPU; use it for relative ordering.
* **BERTScore** uses `distilbert-base-uncased`, not roberta-large.
* **BioBERT licence** is not machine-declared on its hub page; weights are not redistributed.
* **Preprocessing:** the plain-lowercase `raw` strategy won notebook 03 and is what the classical models use (not the regex "clean").
* **Classification label unit:** sentence role, not abstract topic.

## 5. Test results

`python -m pytest -q` → **49 passed** (≈2 min on CPU, with all trained models present: real BioBERT classification, BioBERT NER, T5 summarization, uploads, CSRF, history, PDF/TXT export). An earlier run caught two real bugs, both fixed: an empty-token CSRF bypass, and an upper-cased TXT export title.

## 6. Known issues

* The deployed BioBERT classifier takes ~60 ms/sentence on CPU; a 250-sentence document classifies in ~15 s, plus occlusion explanations for the first 8 sentences.
* The NER model sometimes splits multi-word diseases (e.g. "low" + "back pain") — visible on the Overview specimen.
* Summaries are unverified machine output; the extractive baselines beat T5 on ROUGE (expected for overlap metrics).
* Transformer results are low-resource (1 epoch, ~1.7% of data) and not comparable with published full-data scores.
* PDF extraction needs a text layer (no OCR).

## 7. Final verification

* Full test suite: 49 passed.
* App launched with `python app.py`; every page returns 200; Model Laboratory charts render from `artifacts/evaluation/benchmark.json`; Overview specimen is live BioBERT + NER output.
* Notebooks 01–10 executed with the project venv kernel (Python 3.12.16).
* Not done: responsive layouts were only spot-checked at desktop/tablet widths, not on a phone.

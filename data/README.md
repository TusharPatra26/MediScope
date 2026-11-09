# Data provenance

Raw datasets are **not** committed. `python -m src.data` downloads everything into `data/raw/` (git-ignored). Derived data (`data/processed/`, e.g. Word2Vec, the shared test subset) is also git-ignored.

| Dataset | Used for | Source / download | Format | Verified size | Licence status |
|---|---|---|---|---|---|
| **PubMed 20k RCT** — Dernoncourt & Lee, IJCNLP 2017 | Sentence-role classification (BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS) | `https://github.com/Franck-Dernoncourt/pubmed-rct` → `PubMed_20k_RCT/{train,dev,test}.txt` (≈39 MB) | `###<PMID>` header, then `LABEL<TAB>sentence` lines | train 180,040 sentences / 15,000 abstracts · dev 30,212 / 2,500 · test 30,135 / 2,500 | **No explicit licence.** Derived from MEDLINE/PubMed 2016. The repo README states some abstracts may be protected by copyright and that users must satisfy copyright terms. Research use only; never redistributed. |
| **BC5CDR** — BioCreative V CDR, Li et al., *Database* 2016 (T-NER sentence-split release) | Chemical / Disease NER | `https://huggingface.co/datasets/tner/bc5cdr` → `dataset/{train,valid,test}.json` (≈4 MB) | JSON lines `{tokens, tags}`, integer BIO tags | train 5,228 · dev 5,330 · test 5,865 sentences | Hub card: `license: other`. Underlying corpus from NCBI/BioCreative for research. Not redistributed. |
| **PubMed summarization** — Cohan et al., NAACL 2018, via `ccdv/pubmed-summarization` (`document` config) | Summarization evaluation (article → author abstract) | `https://huggingface.co/datasets/ccdv/pubmed-summarization` → `document/test-*.parquet` (≈58 MB, **test split only**) | Parquet `{article, abstract}` | test 6,658 articles; median article ≈ 2,596 words, median abstract ≈ 213 words | Card declares **no licence**; derived from PubMed Central Open Access. Evaluation only; not redistributed. The 2.2 GB training split is deliberately not downloaded. |

Class distribution, duplicates, missing values, leakage checks and quality findings are computed in `notebooks/01_dataset_understanding.ipynb` (nothing is typed in by hand). Key facts at the time of writing:

* PubMed RCT has no missing values and the official split shares **0 abstracts** between train/dev/test; classes are imbalanced (METHODS 33.0%, RESULTS 32.2%, CONCLUSIONS 15.1%, BACKGROUND 12.1%, OBJECTIVE 7.7% of training sentences).
* Identical sentence strings that carry *different* labels exist (44 distinct strings in train) → an irreducible error floor for context-free classifiers.
* Numbers were kept (the `@`-replaced PubMed variant is not used) because digits help recognise RESULTS sentences.

**Limitations:** randomized-controlled-trial abstracts only; English only; labels reflect authors' structured-abstract headings; BC5CDR covers two entity types only.

**Models attribution:** `distilbert-base-uncased`, `bert-base-uncased`, `Falconsai/medical_summarization` and `t5-small` are Apache-2.0. BioBERT v1.1 (`dmis-lab/biobert-base-cased-v1.1`; Lee et al., *Bioinformatics* 2020) shows no licence field on its hub page — cite the paper and check the original repository before redistributing weights (this project does not).

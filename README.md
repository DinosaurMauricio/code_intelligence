# Code Intelligence

Exploring AI-assisted code understanding.

I documented some notes of my understanding of some key papers: [Overleaf link](https://www.overleaf.com/project/69243c0c9d1a8b982fac51ba).

## Project Overview

This repository contains my exploration of neural code models, specifically focusing on identifier prediction as a task for understanding code semantics and structured data. The work builds on CodeBERT to further fine-tune from CodeSearchNet but only on Python code from to predict masked variable and function names.

While CodeBERT was originally trained with MLM (Masked Language Modeling) and RTD (Replaced Token Detection), here the model is fine-tuned only using MLM on Python code. In the original CodeBERT, code and natural language (NL) components have separate generators, where in this quick work, the model processes both source code and NL (docstrings and comments) together, potentially influencing results. To evaluate this, I experimented with training with and without docstrings.

Given CodeBERT’s original separate generator setup, combining NL and code could affect performance. Using the same training configurations, I trained models with docstrings and without docstrings. Due to the open-ended nature of code prediction, accuracy alone may not be the best metric. Still, results showed a small difference:

| Metric                     | With Comments | Without Comments |
| -------------------------- | ------------- | ---------------- |
| Accuracy                   | 60.75%        | 62.32%           |
| Top-5 Accuracy             | 71.32%        | 72.48%           |
| MRR (Mean Reciprocal Rank) | 64.91%        | 66.36%           |

Overall, the model performed slightly better without comments, likely because CodeBERT’s original training involved two objectives (MLM for code, RTD for NL). Feeding both NL and PL simultaneously may have affected performance, as docstrings, while semantically rich, add complexity that may not always improve prediction for this task.

### TODO

- [x] Implement Top-K accuracy metrics (Top-5)
- [x] Experiment with removing docstrings from input
- [x] Comparison: with vs without docstrings

**Future task. Code Generation:**

- [ ] Build generation task using Decoder (e.g. CodeT5)
- [ ] Implement HumanEval evaluation pipeline
- [ ] Implement Pass@k metrics

## Repository Structure

```
code_refactoring/
├── dataset/
│   ├── code_datasets.py       # Dataset
├── utils/
│   ├── collate.py            # Data collation for batching
│   ├── config.py             # Configuration
│   ├── constants.py          # Project constants
│   ├── data.py               # Data utilities
│   ├── parser.py             # Code parsing
│   ├── processor.py          # Masking logic
│   └── trainer.py            # Training loop and evaluation
├── config.yaml               # Hyperparameters and settings
├── main.py                   # Training entry point
├── dataset_builder.ipynb     # Dataset exploration and creation
├── tree_sitter.ipynb         # AST parsing experiments
├── dataset.parquet           # Processed dataset
├── .gitignore
├── .gitattributes
└── README.md
```

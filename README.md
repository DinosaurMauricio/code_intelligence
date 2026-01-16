# Code Intelligence

Exploring AI-assisted code understanding.

I documented some notes of my understanding of some key papers: [Overleaf link](https://www.overleaf.com/project/69243c0c9d1a8b982fac51ba).

## Project Overview

This repository contains my exploration of neural code models, specifically focusing on identifier prediction as a task for understanding code semantics. The work uses CodeBERT fine-tuned on Python code from CodeSearchNet to predict masked variable and function names.

### TODO

- [x] Implement Top-K accuracy metrics (Top-5)
- [x] Experiment with removing docstrings from input
- [ ] Comparison: with vs without docstrings

**Futere task. Code Generation:**

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

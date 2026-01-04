# Code Intelligence

Exploring AI-assisted code understanding.

I documented some notes of my understanding of some key papers: [Overleaf link](https://www.overleaf.com/project/69243c0c9d1a8b982fac51ba).

**Note**: _Notes include AI-assisted formatting for clarity, with all technical content manually reviewed_

## Project Overview

This repository contains my exploration of neural code models, specifically focusing on identifier prediction as a task for understanding code semantics. The work uses CodeBERT fine-tuned on Python code from CodeSearchNet to predict masked variable and function names.

Initial fine-tuning shows ~1% accuracy improvement over pretrained baseline on test set. However, simple accuracy may not fully capture performance on this task (large vocabulary, multiple valid names possible). Top-K metrics and deeper analysis needed.

### TODO

- [ ] Implement Top-K accuracy metrics (Top-5, Top-10)
- [ ] Experiment with removing docstrings from input
- [ ] Systematic comparison: with vs without docstrings

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
│   ├── trainer.py            # Training loop and evaluation
│   └── __pycache__/
├── config.yaml               # Hyperparameters and settings
├── main.py                   # Training entry point
├── dataset_builder.ipynb     # Dataset exploration and creation
├── tree_sitter.ipynb         # AST parsing experiments
├── dataset.parquet           # Processed dataset
├── .gitignore
├── .gitattributes
└── README.md
```

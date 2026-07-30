# EIGA: Evidence-Guided Information Acquisition

EIGA (Evidence-Guided Information Acquisition) is a research framework for
dependency-aware evidence acquisition under hidden source correlation.

Unlike conventional retrieval systems that treat evidence independently,
EIGA explicitly models hidden dependencies between evidence sources using
Energy-Based Models (EBMs), allowing the acquisition process to distinguish
genuinely informative evidence from redundant or correlated observations.

This repository accompanies the paper:

> **EIGA: Evidence-Guided Information Acquisition under Hidden Evidence Dependency**

---

## Key Features

- Hidden-dependency synthetic benchmark with seven dependency regimes
- Pairwise Evidence Belief Module (Pairwise EBM)
- Latent Evidence Belief Module (Latent EBM)
- PPO-based sequential acquisition policy
- Multi-task supervision
- Reproducible three-seed experiments
- Unit tests and smoke tests

---

## Repository Structure

```text
configs/
data/
results/
scripts/
src/
tests/
paper/
```

---

## Installation

```bash
git clone <repo-url>
cd EIGA

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# or
.venv\Scripts\activate         # Windows

pip install -e .
```

---

## Running Tests

```bash
PYTHONPATH=src pytest
```

Smoke test:

```bash
PYTHONPATH=src python scripts/smoke_test.py
```

---

## Generate Dataset

```bash
python scripts/generate_dataset.py \
    --output data/generated
```

---

## Train Independent EBM

```bash
python scripts/train_independent_ebm.py \
    --data data/generated \
    --output runs/independent_ebm
```

---

## Train Latent EBM

```bash
python scripts/train_latent_ebm.py \
    --data-dir data/generated
```

---

## Run Ablation Study

Quick run:

```bash
python scripts/run_latent_ablations.py \
    --seeds 17 \
    --output-dir results/latent_ablations_quick
```

Full three-seed experiment:

```bash
python scripts/run_latent_ablations.py \
    --seeds 17 42 123 \
    --output-dir results/latent_ablations
```

---

## Main Results

| Method | Accuracy |
|---------|---------:|
| Random | 31.1% |
| Confidence | 52.7% |
| Entropy | 52.7% |
| PPO | 52.6% |
| **Latent EBM** | **84.7%** |
| Oracle | **100%** |

The Latent EBM substantially outperforms conventional acquisition strategies by explicitly modelling hidden evidence dependency.

---

## Paper

The accompanying paper is included in:

```
paper/EIGA_Final_Paper.pdf
```

---

## Reproducibility

All reported experiments are reproducible.

- Fixed random seeds
- Deterministic benchmark generation
- Three-seed ablation studies
- Configuration files included
- Unit tests provided

---

## Citation

```bibtex
@misc{joshi2026eiga,
  title={EIGA: Evidence-Guided Information Acquisition under Hidden Evidence Dependency},
  author={Pratyush Joshi},
  year={2026}
}
```
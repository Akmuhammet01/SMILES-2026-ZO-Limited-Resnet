# SMILES-2026 — Zero-order ResNet18 on CIFAR100 — Solution Report

## Reproducibility

### Environment

- **Python:** 3.12 (example: Conda env `smiles` with dependencies from `requirements.txt`).
- **Packages:** `torch==2.10.0`, `torchvision==0.25.0`, `tqdm==4.67.1` as pinned in `requirements.txt`.
- **Hardware note:** The reference `results.json` in this repository was produced on **CPU** (`torch 2.10.0+cpu`). 

### Commands

From the repository root (after `pip install -r requirements.txt` in your chosen environment):

```bash
python validate.py --data_dir ./data --batch_size 32 --n_batches 32 --output results.json --seed 42
```

- CIFAR100 is downloaded automatically under `--data_dir` on first run.
- The script enforces `n_batches × batch_size ≤ 8192`.
- **`--seed 42`** matches the default in `validate.py` and must be kept if you want the same batch order and pseudo-random behaviour as in the bundled `results.json`.

### Important implementation details

- **No gradients in student files:** `zo_optimizer.py` uses only scalar loss values from `loss_fn()`; there is no `loss.backward()`.
- **Determinism:** `validate.py` calls `seed_everything`, enables deterministic CUDNN behaviour, and sets `torch.use_deterministic_algorithms(True, warn_only=True)`. Training uses `DataLoader(..., generator=generator_train)` with that seed.
- **Forward passes per optimizer step:** Each `ZeroOrderOptimizer.step` calls `loss_fn` **1 + 2 × num_queries** times (one loss at the start of the step, then two evaluations per averaged SPSA query). With the current defaults, `num_queries = 4`, so **9** forward passes per step on the fixed mini-batch.

---

## Final solution description

### Files modified (allowed set only)

| File | Role |
|------|------|
| `zo_optimizer.py` | SPSA gradient estimate, Adam update, layer selection |
| `head_init.py` | Initialization of `fc` (100-class head) |
| `augmentation.py` | Training-time transforms only |
| `train_data.py` | CIFAR100 train split loader wiring and comments |

`validate.py` and `model.py` were **not** changed (per assignment rules).

### Optimizer (`zo_optimizer.py`)

**Estimator:** Simultaneous Perturbation Stochastic Approximation (SPSA) with a **Rademacher** random direction (±1 per coordinate). All active parameters are perturbed together, so each SPSA replicate costs **two** loss evaluations instead of two per scalar parameter (which the skeleton’s per-parameter central difference would require).

**Variance reduction:** `num_queries` independent SPSA directions are averaged on the **same** batch each step (default `4`).

**Update rule:** **Adam** with bias correction on the estimated pseudo-gradients, default learning rate `5e-3`, perturbation scale `eps = 1e-3`.

**Which parameters are tuned:** Only `fc.weight` and `fc.bias`. The rationale is that with a very small step budget, the signal-to-noise ratio of ZO estimates on the full backbone is poor; a linear readout on frozen ImageNet features is the primary lever.

### Head initialization (`head_init.py`)

Weights: `nn.init.normal_(layer.weight, mean=0.0, std=1e-2)`; bias: zeros. Goals: break symmetry between classes (non-degenerate rows) while keeping initial logits small so cross-entropy starts near the uniform baseline instead of at very large values from a wide Kaiming-style head.

### Augmentation (`augmentation.py`)

Training pipeline: resize to 224, random crop with reflect padding, horizontal flip, mild `ColorJitter`, CIFAR100 normalize, then `RandomErasing` on the tensor. Validation pipeline is unchanged.

### Training data (`train_data.py`)

Full CIFAR100 training split, `shuffle=True`, `num_workers=0` for reproducibility with deterministic algorithms on Windows, seeded generator.

### What mattered most (in principle)

- Replacing **per-parameter finite differences** with **SPSA** is the largest structural win: it makes a bounded number of forward passes per step feasible at all.
- **Multi-query SPSA averaging** and **Adam** are aimed at stabilising updates under heavy gradient noise.

---

## Experiments and discarded or weaker directions

1. **Skeleton 2-point central difference per parameter**  
   Correct but unusable at head scale: cost grows linearly with parameter count; abandoned in favour of SPSA.

2. **Heavier augmentation (e.g. AutoAugment / TrivialAugment-wide)**  
   Tried conceptually during design; commented rationale in `augmentation.py`: at a 32-step ZO budget, stronger between-batch augmentation can increase loss variance and slow apparent progress. The shipped pipeline stays moderate.

3. **Tuning ResNet `layer4` or deeper blocks with SPSA**  
   High-dimensional ZO noise and few steps make this hard to justify without many more steps or a different estimator budget; final submission keeps only the head.

4. **Official run outcome vs expectation**  
   On the machine used to generate the committed `results.json`, **fine-tuned top-1 (0.74%) was slightly below initialized-head top-1 (0.89%)**. That indicates the current hyperparameters (`lr`, `eps`, `num_queries`, Adam) are **not yet tuned** for this exact budget and hardware path; SPSA + Adam can diverge or wander when step size or perturbation scale is mis-matched to loss scale. Reasonable next experiments (not left in code as silent magic): lower `lr`, lower `eps`, reduce `num_queries` to allow more optimizer steps under a self-imposed forward budget, or line-search / schedule `eps` across steps.

---

## Results file

The repository includes `results.json` produced by the command in the reproducibility section. Grading metric: **`val_accuracy_top1_finetuned`** (top-1 on the validation set after ZO fine-tuning).

Values in the committed file at the time of writing:

- `val_accuracy_top1_imagenet_head`: 0.0037  
- `val_accuracy_top1_init_head`: 0.0089  
- `val_accuracy_top1_finetuned`: 0.0074  
- `n_batches`: 32, `batch_size`: 32  
- `layers_tuned`: `["fc.weight", "fc.bias"]`  

Re-running `validate.py` with the same seed and environment should reproduce these numbers within the tolerance described in the assignment materials (typically small absolute drift unless floating-point or backend differs).

# Reliability-Aware Federated Learning for Shortcut-Robust Pneumonia Detection in Chest X-rays

> Capstone project, B.M.S. College of Engineering (BMSCE), Bengaluru
> **Team:** Arpita Patel · Anshika Agrawal · Anvita A Dali · Asritha Sriram
> **Guide:** Dr. Sheela S V

A benchmark and black-box audit of **shortcut learning in federated chest X-ray models**. We build a controlled
four-hospital federation from real RSNA images in which two hospitals carry a *real-image* shortcut (view position
or patient sex is correlated with the label), measure how much that hurts the global model, and test a
**reliability-aware aggregation** that down-weights clients whose models rely on the shortcut.

> **Framing.** This is an audit / benchmark / trade-off study, not a claim of state-of-the-art accuracy. We do not
> claim to beat fairness-weighted baselines such as q-FFL in general (see [Results](#results) and
> [Limitations](#limitations)).

---

## Contents
1. [The problem](#the-problem)
2. [Federation design](#federation-design)
3. [Method](#method)
4. [Baselines compared](#baselines-compared)
5. [Results](#results)
6. [Repository layout](#repository-layout)
7. [Getting started](#getting-started)
8. [Default hyper-parameters](#default-hyper-parameters)
9. [Experiments](#experiments)
10. [Limitations](#limitations)
11. [Data and acknowledgements](#data-and-acknowledgements)

---

## The problem

A pneumonia classifier can score well at the hospital where it was trained and still fail elsewhere, because it
learned a **shortcut**: a feature that is correlated with the label *locally* but not causally related to disease.
Example: in RSNA, pneumonia films are mostly portable **AP** (anteroposterior) views of sicker patients
(P(AP | pneumonia) ≈ 0.78 vs P(AP | normal) ≈ 0.36), so "AP view" predicts pneumonia without looking at the lungs.

In federated learning this is worse: a hospital's local validation score looks great *because* of the shortcut,
so naive validation-weighted aggregation can reward the very clients that learned it, and the shortcut leaks
into the shared global model.

**Question:** can each client audit its own model, without sharing data, and can the server use the result to
make the global model more robust when the shortcut flips?

## Federation design

Four simulated hospitals are sampled from the RSNA pneumonia dataset (cue metadata — view, sex, age — comes from the
NIH ChestX-ray14 join). The shortcut is a **sampling bias on real attributes, not pasted pixels**.

| Hospital | Role | Planted cue | Coupling *c* |
|---|---|---|---|
| H1_general | clean | none | 0.5 |
| H2_viewcue | biased | view (AP = 1, PA = 0) | 0.90 |
| H3_ICU | clean | none | 0.5 |
| H4_democue | biased | sex (M = 1, F = 0) | 0.90 |

**Coupling** *c* is P(cue = 1 | pneumonia) = *c* and P(cue = 1 | normal) = 1 − *c*. At *c* = 0.90, H2 has about 90 %
AP among pneumonia films and 90 % PA among normals; H4 has about 90 % male among pneumonia and 90 % female among
normals. *c* = 0.5 means the cue carries no information. We also run *c* = 0.75 and 0.95.

* **Sizes:** 540 train / 135 val images per hospital; three test sets of 420 images.
* **Prevalence:** held at 40 % pneumonia in every hospital so prevalence is never the confound.
* **View is neutralised** (c = 0.5) in every hospital that is not itself view-biased, because pneumonia is naturally
  AP-dominant in RSNA. Sex is naturally about balanced and is left alone elsewhere.
* **Patient-disjoint:** no NIH patient appears in two splits (verified by `verify_disjoint`). Multiple images per
  patient are kept together.
* **Why so small:** sizes are capped by the scarcest (label × cue) cell, pneumonia-positive + PA view (~1.1–1.3k
  images), which is shared by four hospitals and three test sets. Run `python build_federation.py --cells` for the
  exact budget. A separate **dev federation** (15 % of patients) is used only for tuning.

**Three test sets** (all 420 images):

| Test set | Cue coupling | What it shows |
|---|---|---|
| `test_id` | same as training | looks good at home (shortcut helps) |
| `test_balanced` | c = 0.5 | cue carries no information |
| `test_flipped` | 1 − c | the shortcut now misleads (failure set) |

`is_shortcut` ground truth is used **only** by the oracle baseline and for scoring detection. It is never used by
any reliability computation.

## Method

**Model.** DenseNet-121 (ImageNet-pretrained), 224×224 input, single logit, BCE-with-logits loss, dropout 0.2.
Federated rounds follow FedAvg-style delta aggregation: each client trains locally from the global weights, returns
Δᵢ = stateᵢ − global, and the server forms `global + Σ wᵢ Δᵢ / Σ wᵢ`.

**Client audit** (`client_audit.py`, `context_audit.py`). After local training, each client scores its *own* model on its
*private* validation set. Only three scalar signals leave the client:

| Signal | Meaning | Range |
|---|---|---|
| **Cue leakage** ℓ | within one true label, can the model output predict a cue? ℓ = max over cues and labels of 2·\|AUROC(cue ~ p \| Y = y) − 0.5\|, released only for cells with ≥ 20 samples | 0–1, high = shortcut |
| **Subgroup gap** G | overall AUROC minus the worst cue subgroup AUROC | 0–1, high = shortcut |
| **Border-only reliance** ρ | blank the central 50 % with constant gray 0.5; can the model still predict the label from the border alone? ρ = max(0, 2·(AUROC − 0.5)) | 0–1, high = shortcut |

The cue bank (`cue_bank.py`) uses fixed global cut-points, never client-local ones. Cues: view, sex, age band, frame
intensity, brightness.

**Reliability score** (`reliability.py`). Each signal is flipped to *cleanliness* `s = 1 − signal`, so that high means
trustworthy (aggregation weights must grow with trust). The cleanliness scores are fused with a weighted
**geometric mean**, which behaves like an AND: one red flag pulls trust down. The fused score is then

1. **shrunk** toward a neutral prior: `r = λ·r̂ + (1 − λ)·r₀`, with `λ = n_eff / (n_eff + κ)`, κ = 50, r₀ = 0.5.
   Here `n_eff` is the minority-class count of the validation set (≈ 54 for 135 images at 40 % prevalence), so
   λ ≈ 0.52 and r lives in roughly 0.4–0.7;
2. **EMA-smoothed** across rounds (0.5);
3. **warmed up:** the first 3 rounds are plain FedAvg before scores are trusted.

**Aggregation weight:** `wᵢ = nᵢ · max(rᵢ, 0.1)^γ`. γ = 2 is the pre-specified primary setting; γ ∈ {1, 4, 8} are
sensitivity analyses reported as post hoc.

## Baselines compared

`fedavg`, `fedprox` (μ = 0.1), validation-performance weighting (`valperf`), `fedar`, robust update-agreement
(median/sign-agreement) aggregation, `qffl`-style fairness weighting, `oracle_drop` (drops the truly biased clients;
a ceiling, not deployable), and `reliability` (this work), plus centralised and local-only reference runs.

## Results

Main federation, *c* = 0.90, AUROC (higher is better). Run `collate_results.py` to regenerate from `runs/`.

| Method | `test_flipped` | `test_id` |
|---|---|---|
| FedAvg | 0.622 | 0.926 |
| Reliability, γ = 2 (primary) | 0.659 | 0.919 |
| Reliability, γ = 8 | 0.738 | 0.883 |
| Oracle (drops biased clients) | 0.775 | 0.835 |

At *c* = 0.95, FedAvg falls to 0.396 on `test_flipped` (in-distribution 0.964), while γ = 8 reaches 0.646 (0.879) and
the oracle 0.702 (0.832).

**Reading it:** stronger down-weighting (higher γ) buys robustness on the flipped set and pays for it in-distribution,
a **trade-off**, not a free win. Even the oracle trades in-distribution accuracy for robustness. The audit separates
clean from shortcut-driven clients; final r at *c* = 0.90 is H1 0.660, H2 0.546, H3 0.692, H4 0.579. At γ = 2 the
two biased hospitals still carry about 41 % of the aggregation weight, which is why γ = 2 gains are modest.

Figures are in [`figures/`](figures): `FIG*` (paper figures), `P*` (all-methods, second federation, round curves,
r heatmaps, γ trade-off, coupling sweep), `D*` (diagnostics), `F*`/`G*` (shortcut X-rays, Grad-CAM, t-SNE, border mass),
`S*` (schematics), and `P0_summary_table.csv`.

## Repository layout

All code lives at the repository root (flat, `PROJECT_DIR`-relative imports).

| File | Purpose |
|---|---|
| `config.py` | paths, roster, and `Split` / `Train` / `Reliability` / `Privacy` config dataclasses |
| `utils.py` | seeding, device, JSON/CSV IO, logger, numeric helpers |
| `build_metadata.py` | patient metadata table from RSNA + NIH join |
| `build_federation.py` | seeded, patient-disjoint federation with controlled cue coupling; dev federation; cutpoints |
| `hospital_dataset.py`, `hospital_transforms.py` | datasets, augmentation (pixel-artifact generators kept only for the optional appendix) |
| `model.py` | DenseNet-121 wrapper, state clipping |
| `local_train.py` | one client's local training step (FedAvg / FedProx) |
| `local_baseline.py` | independent per-hospital training (baseline and diagnostic) |
| `fed_train.py` | the federated round loop and single-run entry point |
| `aggregate.py` | server-side aggregation rules |
| `cue_bank.py` | pre-registered metadata and image cues with fixed cut-points |
| `client_audit.py`, `context_audit.py` | the local audit signals (leakage, subgroup gap, border-only) |
| `reliability.py` | signal → reliability score, shrinkage, EMA, aggregation weight |
| `evaluate.py` | AUROC / AUPRC / worst-group AUROC / calibration / detection metrics |
| `stats.py` | bootstrap CIs, Wilcoxon, multiple-comparison correction |
| `run_experiments.py` | YAML-driven experiment grid driver |
| `collate_results.py` | `runs/` → tables with mean ± 95 % bootstrap CI and significance tests |
| `explain.py` | Grad-CAM vs behavioural context-ablation analysis |
| `external_eval.py` | zero-shot evaluation on an external dataset |
| `dp.py`, `protocols.py`, `attacks.py` | privacy-protocol mechanics, DP primitives and leakage attacks (exploratory) |
| `synthetic_probe.py` | optional appendix probe (not load-bearing) |
| `make_real_bundle.py` | builds a bundle for the separate `cueaudit` UI (requires that package) |
| `experiments/` | declarative YAML specs E0–E9 |
| `figures/` | result figures (PNG/PDF) |

## Getting started

**Requirements:** Python 3.10+, PyTorch + torchvision, numpy, pandas, scikit-learn, scipy, pyyaml, pydicom/Pillow,
matplotlib. A GPU is strongly recommended (the original runs used Google Colab).

**Data (not included).** Obtain the RSNA Pneumonia Detection dataset from the RSNA website and the NIH ChestX-ray14
`Data_Entry_2017.csv`, and place them as described in the header of `config.py`:

```
PROJECT_DIR/
  RSNA-pneumonia-dataset/
    stage_2_train_images/   stage_2_train_labels.csv   stage_2_detailed_class_info.csv
    pneumonia-challenge-dataset-mappings_2018.json     Data_Entry_2017.csv
  metadata/  federations/  runs/        # created by the scripts
```

```bash
export PROJECT_DIR=/path/to/capstone_project      # defaults to the Colab Drive path

python build_metadata.py                           # 1. patient metadata
python build_federation.py --cells                 # 2a. inspect the cell budget
python build_federation.py                         # 2b. build the main federation (c = 0.90)
python build_federation.py --coupling 0.95 --name c095

python run_experiments.py experiments/E1_main.yaml # 3. run an experiment grid
python collate_results.py                          # 4. tables and significance tests
```

A single federated run is available through `fed_train.py` (see `--help`).

## Default hyper-parameters

From `config.py` (`TrainConfig`): batch size 32, Adam, learning rate 1e-4, weight decay 1e-4, dropout 0.2, mixed
precision, **1 local epoch per round**. Rounds, seeds and methods per experiment are set in the YAML files; check
`experiments/*.yaml` and the saved run config in `runs/` for the exact values used in any reported result.

*Round vs epoch:* a **round** is one federated cycle (server broadcast → local training → return updates →
aggregate). A **local epoch** is one pass over a client's own training data inside a round.

## Experiments

| Spec | Purpose |
|---|---|
| `E0_pilot` | motivating local-only diagnostic and pilot |
| `E1_main` | headline comparison of all aggregation methods at c = 0.90 |
| `E2_coupling_sweep` | robustness as the planted coupling varies |
| `E3_do_no_harm` | behaviour when no client is biased |
| `E6_ablations` | signal and component ablations |
| `E8_privacy` | privacy protocols and DP trade-offs (exploratory) |
| `E9`, `E9b` | independent second federation draw (γ = 2 and γ = 8) |

## Limitations

* **Small validation sets.** Each audit uses 135 images (n_eff ≈ 54), so scores are noisy; shrinkage, EMA and warm-up
  compensate but cannot remove this. Reliability values only span about 0.4–0.7.
* **Simulated federation.** Shortcuts are planted by sampling real images; the biased hospitals are the
  high-AUROC clients, which also helps AUROC-based fairness weighting such as q-FFL for reasons unrelated to
  shortcut detection.
* **Detection estimates are optimistic** where federation draws are shared across folds.
* **Planted cue direction.** For view the planted direction matches nature (AP ↔ pneumonia); for sex it is arbitrary.
* **One dataset family** (RSNA/NIH). External generalisation is limited (`external_eval.py`).
* **Research prototype. Not a medical device and not for clinical use.**

## Data and acknowledgements

* RSNA Pneumonia Detection Challenge data, obtained from the RSNA website; cue metadata from NIH ChestX-ray14.
  Follow the original data-use terms; the data is **not** redistributed here.
* DenseNet-121 with ImageNet pre-training (torchvision).
* Guide: Dr. Sheela S V, BMSCE.

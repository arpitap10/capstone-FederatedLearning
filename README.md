# Reliability-Aware Federated Learning for Shortcut-Robust Pneumonia Detection in Chest X-rays

Capstone project, BMSCE (Dr. Sheela S V). A benchmark and audit of shortcut
learning in federated chest X-ray models on RSNA pneumonia data. Four simulated
hospitals: H1/H3 clean, H2 view-cue biased (AP/PA), H4 sex-cue biased. Black-box
audit signals (cue leakage, subgroup gap, border-only reliance) are fused into a
per-client reliability score that weights FedAvg aggregation.

## Run order
1. `build_metadata.py` - patient metadata (RSNA/NIH join)
2. `build_federation.py` - seeded, patient-disjoint federation with planted cues
3. `fed_train.py` / `run_experiments.py` - train and compare methods (configs in `experiments/`)
4. `collate_results.py`, then `make_*_figures.py` - tables and figures (outputs in `figures/`)

Set `PROJECT_DIR` (see `config.py`). The RSNA data and trained checkpoints are not included.
Research prototype, not for clinical use.

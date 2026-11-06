"""
run_experiments.py - YAML-driven experiment grid driver (plan sections 9-14).

This is the single entry point that turns one declarative YAML spec into a batch
of `fed_train.run_federation` calls, a tidy per-run results CSV, and a
paper-ready summary table (mean +/- bootstrap CI per method, plus a Wilcoxon
signed-rank test of every method against FedAvg).  It never trains a model
itself: it only *builds configs*, *builds federations if missing*, *dispatches
runs*, and *aggregates results*.

----------------------------------------------------------------------
Mapping to the experiment plan (E0-E11)
----------------------------------------------------------------------
Each experiment in the plan is expressed as a YAML file under
``src/experiments/``.  The ones shipped here:

  E0  pilot / go-no-go     - FedAvg on the default roster; confirm the shortcut
                             actually hurts (large drop on test_flipped).
  E1  main comparison      - all 8 methods, 10 seeds, 20 rounds, protocol P1.
                             Primary endpoint table + Wilcoxon vs fedavg.
  E2  coupling sweep       - fedavg/reliability/oracle_drop across coupling
                             c in {0.6,0.8,0.95}; one federation+grid per c.
  E3  do-no-harm           - m=0 (no biased hospital); reliability must not hurt
                             a clean federation relative to fedavg.
  E6  ablations            - reliability only; sweep gamma and knock out each
                             fused signal (w_leak/w_gap/w_ctx/w_tau -> 0).
  E8  privacy              - reliability under score-DP and update-DP noise
                             (score_dp_sigma, update_clip_C/update_dp_sigma).

E4/E5/E7/E9-E11 (extra rosters, P2 transfer, external Tier-C, qualitative) are
not shipped as spec files but need no driver changes: write another YAML using
the schema below (e.g. a P2 run is just ``protocol: P2`` with a privacy block,
a 3-biased-hospital roster is an ``m: 3`` style federation once the roster in
config.py is extended).

Experiments are NOT run automatically - you point the driver at one file:

    python run_experiments.py --config experiments/E1_main.yaml
    python run_experiments.py --config experiments/E2_coupling_sweep.yaml --dry-run

----------------------------------------------------------------------
YAML schema
----------------------------------------------------------------------
Top level keys (``*`` = required):

  exp_name*        str    short id used in output filenames and the CSV.
  federation*      map    base federation:
                            name*     str    federation directory under
                                             federations/ (dev twin is name_dev)
                            coupling  float  coupling c for biased hospitals
                            m         int    number of biased hospitals to keep
                                             (0 disables all -> clean federation)
                            seed      int    federation draw seed (SplitConfig.seed)
  methods*         list   subset of: fedavg, fedprox, valperf, fedar, qffl,
                            update_agreement, oracle_drop, reliability.
  seeds*           list   training seeds (one run per method x seed x arm).
  rounds           int    federated rounds (default: TrainConfig.rounds).
  protocol         str    P0|P1|P2|P3 -> PrivacyConfig.protocol (default P1).
  overrides        map    base config overrides, keyed by the CONFIG_REGISTRY
                          groups ``train`` / ``reliability`` / ``privacy``.  Each
                          sub-map's keys must be real dataclass fields (unknown
                          keys raise a clear error).
  sweeps           list   optional list of named "arms".  Each arm is run as its
                          own method x seed grid and may override:
                            name*        str  label for the arm (-> CSV + tag)
                            federation   map  partial federation overrides
                                              (e.g. a different coupling); if it
                                              changes the draw but gives no
                                              ``name`` a suffix ``_<arm>`` is
                                              appended so federations do not clash
                            overrides    map  train/reliability/privacy overrides
                                              deep-merged over the base overrides
                          With no ``sweeps`` the spec runs a single implicit arm.

----------------------------------------------------------------------
Outputs (under runs/_experiments/<exp_name>/)
----------------------------------------------------------------------
  results.csv      one row per run: experiment, arm, fed, method, seed, protocol,
                   status, error, + flattened `final` round metrics + detection
                   metrics.  Individual run artefacts stay under runs/<out_tag>/.
  summary.csv      per (arm, method) mean + bootstrap CI of the endpoints.
  summary.json     the same, plus the Wilcoxon-vs-fedavg table and the resolved
                   spec, for programmatic consumption / the paper tables.

Primary endpoint   : test_balanced worst-group AUROC  (test_balanced_wg_auroc)
Secondary endpoints: test_flipped AUROC (test_flipped_auroc),
                     flip_drop, detection AUROC (detection_auroc).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import traceback
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

try:
    import yaml
except ImportError as e:  # pragma: no cover - surfaced clearly at call time
    raise ImportError("run_experiments.py needs pyyaml: pip install pyyaml") from e

import config
from config import SplitConfig, CONFIG_REGISTRY
from utils import get_logger, save_json

LOG = get_logger()

# Methods the aggregator understands (keep in sync with aggregate.py / fed_train).
VALID_METHODS = ("fedavg", "fedprox", "valperf", "fedar", "qffl",
                 "update_agreement", "oracle_drop", "reliability")
BASELINE_METHOD = "fedavg"

# Endpoints reported in the summary table.  Keys match the flattened `final`
# round metrics / detection metrics produced by run_federation.
PRIMARY_ENDPOINT = "test_balanced_wg_auroc"
SECONDARY_ENDPOINTS = ("test_flipped_auroc", "flip_drop", "detection_auroc")
SUMMARY_ENDPOINTS = (PRIMARY_ENDPOINT,) + SECONDARY_ENDPOINTS

EXPERIMENTS_OUT = os.path.join(config.RUNS_DIR, "_experiments")


# ======================================================================
# Spec resolution
# ======================================================================
class Arm:
    """One method x seed grid: a resolved federation spec + config overrides."""

    def __init__(self, name: str, fed: Dict[str, Any], overrides: Dict[str, Any]):
        self.name = name
        self.fed = fed
        self.overrides = overrides

    @property
    def fed_name(self) -> str:
        return self.fed["name"]


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `extra` over `base` (used for the override groups)."""
    out = copy.deepcopy(base) if base else {}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_arms(spec: Dict[str, Any]) -> List[Arm]:
    """Expand a spec into one or more arms (one per sweep entry, else one)."""
    base_fed = dict(spec["federation"])
    base_over = spec.get("overrides", {}) or {}
    sweeps = spec.get("sweeps")
    if not sweeps:
        return [Arm("base", base_fed, base_over)]

    arms: List[Arm] = []
    seen_names = set()
    for sw in sweeps:
        if "name" not in sw:
            raise ValueError("every sweep entry needs a 'name'")
        arm_name = str(sw["name"])
        if arm_name in seen_names:
            raise ValueError(f"duplicate sweep name {arm_name!r}")
        seen_names.add(arm_name)
        fed = dict(base_fed)
        fed_override = sw.get("federation", {}) or {}
        fed.update(fed_override)
        # If the draw changed but no explicit name was given, avoid clobbering
        # the base federation by suffixing the arm name.
        changes_draw = any(k in fed_override for k in ("coupling", "m", "seed"))
        if changes_draw and "name" not in fed_override:
            fed["name"] = f"{base_fed['name']}_{arm_name}"
        over = _deep_merge(base_over, sw.get("overrides", {}))
        arms.append(Arm(arm_name, fed, over))
    return arms


def validate_spec(spec: Dict[str, Any]) -> None:
    for key in ("exp_name", "federation", "methods", "seeds"):
        if key not in spec:
            raise ValueError(f"spec missing required key {key!r}")
    if "name" not in spec["federation"]:
        raise ValueError("federation block needs a 'name'")
    bad = [m for m in spec["methods"] if m not in VALID_METHODS]
    if bad:
        raise ValueError(f"unknown method(s) {bad}; valid: {VALID_METHODS}")
    proto = spec.get("protocol", "P1")
    if proto not in ("P0", "P1", "P2", "P3"):
        raise ValueError(f"protocol must be P0..P3, got {proto!r}")


# ======================================================================
# Config construction
# ======================================================================
def build_config(group: str, overrides: Dict[str, Any]):
    """Instantiate a registry dataclass from its defaults + YAML overrides.

    Raises KeyError with a clear message on any field that is not a real
    attribute of the dataclass, so typos in the YAML fail fast.
    """
    if group not in CONFIG_REGISTRY:
        raise KeyError(f"unknown config group {group!r}; "
                       f"valid groups: {sorted(CONFIG_REGISTRY)}")
    cls = CONFIG_REGISTRY[group]
    cfg = cls()
    valid = {f.name for f in dataclasses.fields(cls)}
    for k, v in (overrides or {}).items():
        if k not in valid:
            raise KeyError(f"unknown {group} config key {k!r}; "
                           f"valid keys: {sorted(valid)}")
        setattr(cfg, k, v)
    return cfg


def configs_for_run(arm: Arm, method: str, protocol: str):
    """Fresh (tcfg, rcfg, pcfg) for one run so arms/methods never share state."""
    over = arm.overrides
    tcfg = build_config("train", over.get("train", {}))
    rcfg = build_config("reliability", over.get("reliability", {}))
    pcfg = build_config("privacy", over.get("privacy", {}))
    # Protocol from the spec unless the privacy override set it explicitly.
    if "protocol" not in over.get("privacy", {}):
        pcfg.protocol = protocol
    # FedProx needs a positive proximal term; honour an explicit override.
    if method == "fedprox" and tcfg.fedprox_mu <= 0:
        tcfg.fedprox_mu = 0.1
    return tcfg, rcfg, pcfg


def split_config_for_fed(fed: Dict[str, Any]) -> SplitConfig:
    """Derive a SplitConfig matching build_federation's CLI semantics."""
    cfg = SplitConfig()
    if fed.get("seed") is not None:
        cfg.seed = int(fed["seed"])
    coupling = fed.get("coupling")
    if coupling is not None:
        for v in cfg.roster.values():
            if v["is_shortcut"]:
                v["coupling"] = float(coupling)
    m = fed.get("m")
    if m is not None:
        biased = [h for h, v in cfg.roster.items() if v["is_shortcut"]]
        for h in biased[int(m):]:
            cfg.roster[h] = {"cue": None, "coupling": 0.5, "is_shortcut": False}
    return cfg


# ======================================================================
# Federation building (idempotent)
# ======================================================================
def federation_exists(name: str) -> bool:
    return os.path.exists(os.path.join(config.FEDERATIONS_DIR, name,
                                       "hospital_manifest.json"))


def ensure_federation(fed: Dict[str, Any], dry_run: bool = False) -> None:
    """Build the federation (and its _dev twin) if its manifest is missing."""
    name = fed["name"]
    if federation_exists(name):
        LOG.info(f"[fed] {name}: present, skipping build")
        return
    if dry_run:
        LOG.info(f"[fed] {name}: MISSING -> would build "
                 f"(coupling={fed.get('coupling')}, m={fed.get('m')}, "
                 f"seed={fed.get('seed')})")
        return
    LOG.info(f"[fed] {name}: MISSING -> building "
             f"(coupling={fed.get('coupling')}, m={fed.get('m')}, "
             f"seed={fed.get('seed')})")
    from build_federation import build_federation
    build_federation(split_config_for_fed(fed), name=name)


# ======================================================================
# Result flattening
# ======================================================================
def flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Pull `final` round metrics + scalar detection metrics into a flat row."""
    row: Dict[str, Any] = {}
    for k, v in (summary.get("final") or {}).items():
        if isinstance(v, (int, float)):
            row[k] = v
    det = summary.get("detection") or {}
    for k, v in det.items():
        if isinstance(v, (int, float, bool)):
            row[f"detection_{k}" if not k.startswith("detection") else k] = v
    return row


def _planned_tag(exp_name: str, arm: Arm, method: str, seed: int) -> str:
    label = exp_name if arm.name == "base" else f"{exp_name}.{arm.name}"
    return f"{label}__{arm.fed_name}__{method}__s{seed}"


# ======================================================================
# Grid execution
# ======================================================================
def run_grid(spec: Dict[str, Any], dry_run: bool = False) -> pd.DataFrame:
    exp_name = spec["exp_name"]
    methods = list(spec["methods"])
    seeds = list(spec["seeds"])
    rounds = spec.get("rounds")
    protocol = spec.get("protocol", "P1")
    arms = resolve_arms(spec)

    LOG.info(f"experiment {exp_name}: {len(arms)} arm(s) x {len(methods)} "
             f"method(s) x {len(seeds)} seed(s) = "
             f"{len(arms) * len(methods) * len(seeds)} runs")

    # Build any missing federations first (idempotent, one per distinct name).
    for name in sorted({a.fed_name for a in arms}):
        fed = next(a.fed for a in arms if a.fed_name == name)
        ensure_federation(fed, dry_run=dry_run)

    rows: List[Dict[str, Any]] = []
    for arm in arms:
        for method in methods:
            for seed in seeds:
                tag = _planned_tag(exp_name, arm, method, seed)
                base_row = {"experiment": exp_name, "arm": arm.name,
                            "fed": arm.fed_name, "method": method,
                            "seed": seed, "protocol": protocol, "tag": tag}
                if dry_run:
                    LOG.info(f"[plan] {tag}")
                    rows.append({**base_row, "status": "planned"})
                    continue
                rows.append(_execute_run(spec, arm, method, seed, protocol,
                                         rounds, tag, base_row))
    return pd.DataFrame(rows)


def _execute_run(spec, arm, method, seed, protocol, rounds, tag, base_row):
    """Run one (arm, method, seed); never raise - record failures and continue."""
    from fed_train import run_federation
    try:
        tcfg, rcfg, pcfg = configs_for_run(arm, method, protocol)
        LOG.info(f"[run] {tag}")
        summary = run_federation(arm.fed_name, method=method, tcfg=tcfg,
                                 rcfg=rcfg, pcfg=pcfg, rounds=rounds,
                                 seed=seed, out_tag=tag)
        return {**base_row, "status": "ok", "error": "",
                **flatten_summary(summary)}
    except Exception as e:  # noqa: BLE001 - one bad seed must not kill the grid
        LOG.error(f"[run] {tag} FAILED: {e}")
        LOG.debug(traceback.format_exc())
        return {**base_row, "status": "failed", "error": repr(e)}


# ======================================================================
# Summarisation
# ======================================================================
def summarize(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Per (arm, method) means+CIs and a Wilcoxon-vs-fedavg table per arm."""
    from stats import summarize_runs, wilcoxon_vs_baseline

    ok = df[df["status"] == "ok"].copy() if "status" in df else df.copy()
    metrics = [m for m in SUMMARY_ENDPOINTS if m in ok.columns]
    summary_table = pd.DataFrame()
    if not ok.empty and metrics:
        summary_table = summarize_runs(ok, ["arm", "method"], metrics)

    # Wilcoxon of each method against fedavg on the primary endpoint, per arm.
    wilcoxon: Dict[str, Any] = {}
    if PRIMARY_ENDPOINT in ok.columns:
        for arm_name, sub in ok.groupby("arm"):
            if BASELINE_METHOD not in set(sub["method"]):
                continue
            seeds = sorted(sub["seed"].unique())
            scores: Dict[str, List[float]] = {}
            for method, msub in sub.groupby("method"):
                by_seed = msub.set_index("seed")[PRIMARY_ENDPOINT]
                scores[method] = [float(by_seed.get(s, float("nan")))
                                  for s in seeds]
            if len(scores) < 2:
                continue
            try:
                wilcoxon[arm_name] = wilcoxon_vs_baseline(scores, BASELINE_METHOD)
            except Exception as e:  # noqa: BLE001
                LOG.warning(f"wilcoxon skipped for arm {arm_name}: {e}")
    return {
        "exp_name": spec["exp_name"],
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoints": list(SECONDARY_ENDPOINTS),
        "spec": spec,
        "table": summary_table.to_dict(orient="records") if not summary_table.empty else [],
        "wilcoxon_vs_fedavg": wilcoxon,
    }, summary_table


# ======================================================================
# Main
# ======================================================================
def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    validate_spec(spec)
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", required=True, help="path to an experiment YAML")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned federations and runs, execute nothing")
    args = ap.parse_args()

    spec = load_spec(args.config)
    config.ensure_dirs()
    out_dir = os.path.join(EXPERIMENTS_OUT, spec["exp_name"])
    os.makedirs(out_dir, exist_ok=True)

    df = run_grid(spec, dry_run=args.dry_run)

    if args.dry_run:
        LOG.info(f"[dry-run] {len(df)} planned run(s); nothing executed")
        return

    results_csv = os.path.join(out_dir, "results.csv")
    df.to_csv(results_csv, index=False)
    LOG.info(f"wrote {results_csv} ({len(df)} rows)")

    n_ok = int((df["status"] == "ok").sum()) if "status" in df else 0
    n_fail = int((df["status"] == "failed").sum()) if "status" in df else 0
    LOG.info(f"runs: {n_ok} ok, {n_fail} failed")

    summary_json, summary_table = summarize(df, spec)
    if not summary_table.empty:
        summary_csv = os.path.join(out_dir, "summary.csv")
        summary_table.to_csv(summary_csv, index=False)
        LOG.info(f"wrote {summary_csv}")
    save_json(summary_json, os.path.join(out_dir, "summary.json"))
    LOG.info(f"wrote {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()

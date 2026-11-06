"""
build_federation.py - construct a seeded, patient-disjoint federation with
controlled, real-image cue-label coupling.

Coupling convention (plan 6.3):   P(cue=1 | Y=1) = c ,   P(cue=1 | Y=0) = 1-c.
c = 0.5 means the cue is independent of the label (balanced).

What is built (under federations/<name>/):
  hospitals/<H>/train.csv , val.csv     one pool per client; val = V_i (private)
  tests/test_id.csv                     coupling preserved (looks good at home)
  tests/test_balanced.csv               every eval cue balanced (c=0.5)
  tests/test_flipped.csv                every planted cue flipped (failure set)
  hospital_manifest.json                cue, coupling, prevalence, n, is_shortcut
  cue_cutpoints.json (metadata/)        global cutpoints for image cues

Disjointness is enforced at the NIH-PATIENT level (RSNA is an NIH subset; one
NIH patient may appear under several RSNA ids).  A global "claimed" set of NIH
ids guarantees no patient appears in two splits.  A separate DEV federation is
built from a disjoint pool for all hyper-parameter/threshold tuning.

Clean hospitals balance every eval cue at 0.5; a biased hospital sets ITS cue to
its coupling and balances the other eval cues - so each planted shortcut is
isolated to one cue.

Usage
-----
    python build_federation.py                       # default roster, name "main"
    python build_federation.py --coupling 0.8 --name c080
    python build_federation.py --m 1                 # only one biased hospital
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import config
from config import SplitConfig, ensure_dirs
from utils import get_logger, save_json, set_seed

LOG = get_logger()

_CUE_MAP = {
    "view": {"AP": 1, "PA": 0},
    "sex": {"M": 1, "F": 0},
}


# ----------------------------------------------------------------------
# cue value extraction (metadata-only cues; matches cue_bank definitions)
# ----------------------------------------------------------------------
def cue_series(df: pd.DataFrame, cue: str) -> pd.Series:
    """Float series of a metadata cue: 1.0 / 0.0 / NaN."""
    if cue in _CUE_MAP:
        return df[cue].map(_CUE_MAP[cue]).astype("float")
    if cue == "age_band":
        v = (df["age"] >= 65).astype("float")
        v[df["age"].isna()] = np.nan
        return v
    raise ValueError(f"unsupported coupling cue '{cue}' (metadata cues only)")


# ----------------------------------------------------------------------
# cell plan + sampling
# ----------------------------------------------------------------------
def _p_cue_given_y(val: int, y: int, c: float) -> float:
    p1 = c if y == 1 else (1.0 - c)      # P(cue=1 | y)
    return p1 if val == 1 else (1.0 - p1)


def build_cell_plan(n: int, prevalence: float,
                    cue_couplings: Dict[str, float]) -> Dict[Tuple, int]:
    """Target count for every (y, cue1=v1, cue2=v2, ...) cell under conditional
    independence of cues given y."""
    cues = sorted(cue_couplings)
    plan: Dict[Tuple, int] = {}
    for y in (0, 1):
        py = prevalence if y == 1 else (1.0 - prevalence)
        for combo in product((0, 1), repeat=len(cues)):
            p = py
            for cue, val in zip(cues, combo):
                p *= _p_cue_given_y(val, y, cue_couplings[cue])
            cell = (("y", y),) + tuple((cue, v) for cue, v in zip(cues, combo))
            plan[cell] = int(round(n * p))
    return plan


def sample_to_plan(pool: pd.DataFrame, plan: Dict[Tuple, int], cues: List[str],
                   rng: np.random.Generator, claimed: Set[str],
                   group_by_patient: bool = True) -> pd.DataFrame:
    """Draw images matching each cell, keeping NIH-patient disjointness.

    group_by_patient=True (default): a patient is assigned WHOLLY to one draw -
    all of a selected patient's images may be used (within this draw), and once
    any image of a patient is taken, that patient is locked out of every LATER
    draw.  This keeps the leakage guarantee at the patient level while recovering
    the full image budget (RSNA has ~2.3 images per NIH patient).

    group_by_patient=False: the old behaviour, one image per patient everywhere.

    `claimed` holds patients already used by EARLIER consumers (hard-excluded).
    Patients first used inside this draw are reusable across cells of this same
    draw, then committed to `claimed` at the end.  Warns (does not crash) on a
    short cell.
    """
    work = pool.copy()
    for cue in cues:
        work[f"_c_{cue}"] = cue_series(work, cue)
    nid = work["nih_patient_id"].astype(str)
    work = work[~nid.isin(claimed)].copy()           # drop other consumers' patients

    picked_idx: List[int] = []
    local: Set[str] = set()                           # patients claimed by THIS draw
    shortfalls = []
    for cell, count in plan.items():
        if count <= 0:
            continue
        y = cell[0][1]
        mask = (work["Target"] == y)
        for cue, val in cell[1:]:
            mask &= (work[f"_c_{cue}"] == val)
        cand = work[mask]
        cand = cand[~cand.index.isin(picked_idx)]     # not already taken this draw
        if not group_by_patient:
            cand = cand.drop_duplicates("nih_patient_id")
        if len(cand) < count:
            shortfalls.append((cell, count, len(cand)))
            take = cand
        else:
            take = cand.sample(n=count, random_state=int(rng.integers(1 << 31)))
        local.update(take["nih_patient_id"].astype(str).tolist())
        picked_idx.extend(take.index.tolist())

    claimed.update(local)                             # lock this draw's patients
    if shortfalls:
        for cell, want, have in shortfalls:
            LOG.warning(f"  cell {cell}: wanted {want}, had {have}")
    out = pool.loc[picked_idx].copy()
    return out.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


# ----------------------------------------------------------------------
# roster helpers
# ----------------------------------------------------------------------
def eval_cues_of(roster: Dict[str, Dict]) -> List[str]:
    return sorted({v["cue"] for v in roster.values() if v["cue"]})


def planted_coupling_by_cue(roster: Dict[str, Dict]) -> Dict[str, float]:
    """Per-cue planted coupling = max coupling across biased hospitals on it."""
    out: Dict[str, float] = {}
    for v in roster.values():
        if v["cue"] and v["is_shortcut"]:
            out[v["cue"]] = max(out.get(v["cue"], 0.0), v["coupling"])
    return out


def control_cues_of(roster: Dict[str, Dict], balance_cue: Optional[str]) -> List[str]:
    """Cues the TEST sets control: the biased cues plus the natural-confound cue."""
    cues = set(eval_cues_of(roster))
    if balance_cue:
        cues.add(balance_cue)
    return sorted(cues) if cues else (["view"] if balance_cue is None else [balance_cue])


def hospital_cue_couplings(spec: Dict, balance_cue: Optional[str]) -> Dict[str, float]:
    """Coupling targets for ONE hospital.

    A hospital controls only two things, which keeps it off the scarce joint
    cells (e.g. positive+female+PA):
      (1) the natural-confound cue `balance_cue` (view) -> balanced at 0.5, UNLESS
          this hospital is itself biased on that cue;
      (2) its own planted cue -> at its coupling.
    Any other cue (e.g. sex for a clean or view-biased hospital) is left to its
    natural distribution, which for sex is already ~balanced in RSNA.
    """
    cc: Dict[str, float] = {}
    biased_on = spec["cue"] if (spec["cue"] and spec["is_shortcut"]) else None
    if balance_cue and biased_on != balance_cue:
        cc[balance_cue] = 0.5
    if biased_on:
        cc[biased_on] = spec["coupling"]
    return cc


# ----------------------------------------------------------------------
# core build (one pool -> one federation dir)
# ----------------------------------------------------------------------
_COLS = ["patientId", "nih_patient_id", "Target", "view", "sex", "age", "age_band", "class"]


def report_pool_cells(pool: pd.DataFrame, eval_cues: List[str],
                      prevalence: float = 0.40,
                      group_by_patient: bool = True) -> pd.DataFrame:
    """Per-(Target, cue) cell availability in a pool.

    With group_by_patient=True the sampling unit is the IMAGE (patients disjoint
    across splits, all images kept) so the budget is image-level; otherwise it is
    the deduplicated PATIENT.  `cap_c*` is the largest matched-prevalence size the
    scarcest cell allows IF it were the only consumer - the builder shares each
    cell across all hospitals + 3 test sets, so a usable per-split size is roughly
    cap / (num_hospitals + 3).  Patients_total is shown for context.
    """
    rows = []
    for cue in eval_cues:
        cv = cue_series(pool, cue)
        sub = pool.assign(_cue=cv).dropna(subset=["_cue"])
        unit = sub if group_by_patient else sub.drop_duplicates("nih_patient_id")
        y = unit["Target"].to_numpy()
        c = unit["_cue"].to_numpy()
        cells = {(yy, cc): int(np.sum((y == yy) & (c == cc)))
                 for yy in (0, 1) for cc in (0, 1)}
        caps = {}
        for cpl in (0.5, 0.9):
            fr = {(1, 1): cpl * prevalence, (1, 0): (1 - cpl) * prevalence,
                  (0, 1): (1 - cpl) * (1 - prevalence), (0, 0): cpl * (1 - prevalence)}
            caps[cpl] = int(min(cells[k] / fr[k] if fr[k] > 0 else np.inf for k in cells))
        rows.append({"cue": cue,
                     "images_total": int(len(sub)),
                     "patients_total": int(sub["nih_patient_id"].nunique()),
                     "pos_cue1": cells[(1, 1)], "pos_cue0": cells[(1, 0)],
                     "neg_cue1": cells[(0, 1)], "neg_cue0": cells[(0, 0)],
                     "cap_c0.5": caps[0.5], "cap_c0.9": caps[0.9]})
    rep = pd.DataFrame(rows)
    unit_name = "IMAGE (group_by_patient)" if group_by_patient else "PATIENT (dedup)"
    LOG.info(f"per-cell availability, sampling unit = {unit_name}:\n"
             + rep.to_string(index=False))
    LOG.info("cap_c* assumes ONE consumer; the builder shares each cell across "
             "all hospitals + 3 test sets -> usable per-split ~ cap/(K+3).")
    return rep


def build_one(pool: pd.DataFrame, out_dir: str, cfg: SplitConfig,
              rng: np.random.Generator, scale: float = 1.0) -> Dict:
    roster = cfg.roster
    ecues = eval_cues_of(roster)                       # biased cues (for the manifest)
    ctrl = control_cues_of(roster, cfg.balance_cue)    # cues the TEST sets control
    n_train = max(1, int(round(cfg.train_per_hospital * scale)))
    n_val = max(1, int(round(cfg.val_per_hospital * scale)))
    n_test = max(1, int(round(cfg.test_size * scale)))
    claimed: Set[str] = set()
    os.makedirs(os.path.join(out_dir, "hospitals"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tests"), exist_ok=True)
    manifest = {"eval_cues": ecues, "control_cues": ctrl,
                "balance_cue": cfg.balance_cue, "prevalence": cfg.prevalence,
                "sizes": {"train": n_train, "val": n_val, "test": n_test},
                "hospitals": {}}

    # --- test sets first (claim scarce cells before hospitals) ---
    planted = planted_coupling_by_cue(roster)
    id_couplings = {c: planted.get(c, 0.5) for c in ctrl}
    bal_couplings = {c: 0.5 for c in ctrl}
    flip_couplings = {c: 1.0 - planted.get(c, 0.5) for c in ctrl}

    for name, cc in (("test_id", id_couplings), ("test_balanced", bal_couplings),
                     ("test_flipped", flip_couplings)):
        plan = build_cell_plan(n_test, cfg.prevalence, cc)
        df = sample_to_plan(pool, plan, ctrl, rng, claimed, cfg.group_by_patient)
        df[_COLS].to_csv(os.path.join(out_dir, "tests", f"{name}.csv"), index=False)
        LOG.info(f"  {name}: {len(df)} images (couplings {cc})")

    # --- hospitals, most-constrained first ---
    # A hospital controlling more cues (e.g. a sex-biased hospital that also
    # balances view) competes for scarcer joint cells, so it must draw before the
    # single-cue clean hospitals, which can absorb whatever remains.  Biased
    # hospitals break ties ahead of clean ones.
    def _constraint_rank(h):
        s = roster[h]
        return (-len(hospital_cue_couplings(s, cfg.balance_cue)), not s["is_shortcut"])
    build_order = sorted(cfg.hospitals, key=_constraint_rank)
    LOG.info(f"  hospital build order (most-constrained first): {build_order}")

    for hname in build_order:
        spec = roster[hname]
        cc = hospital_cue_couplings(spec, cfg.balance_cue)
        cues_h = sorted(cc) if cc else []
        hdir = os.path.join(out_dir, "hospitals", hname)
        os.makedirs(hdir, exist_ok=True)
        train_plan = build_cell_plan(n_train, cfg.prevalence, cc)
        val_plan = build_cell_plan(n_val, cfg.prevalence, cc)
        train_df = sample_to_plan(pool, train_plan, cues_h, rng, claimed, cfg.group_by_patient)
        val_df = sample_to_plan(pool, val_plan, cues_h, rng, claimed, cfg.group_by_patient)
        train_df[_COLS].to_csv(os.path.join(hdir, "train.csv"), index=False)
        val_df[_COLS].to_csv(os.path.join(hdir, "val.csv"), index=False)
        manifest["hospitals"][hname] = {
            "cue": spec["cue"], "coupling": spec["coupling"],
            "is_shortcut": bool(spec["is_shortcut"]),
            "cue_couplings": cc, "prevalence": cfg.prevalence,
            "n_train": int(len(train_df)), "n_val": int(len(val_df)),
            "train_pos": int(train_df["Target"].sum()),
        }
        LOG.info(f"  {hname}: train={len(train_df)} val={len(val_df)} "
                 f"cue={spec['cue']} c={spec['coupling']} shortcut={spec['is_shortcut']}")

    save_json(manifest, os.path.join(out_dir, "hospital_manifest.json"))
    return manifest


# ----------------------------------------------------------------------
# disjointness verification
# ----------------------------------------------------------------------
def verify_disjoint(fed_dir: str) -> None:
    """Hard assert: no NIH patient appears in two splits of this federation."""
    seen: Dict[str, str] = {}
    csvs = []
    hdir = os.path.join(fed_dir, "hospitals")
    for h in sorted(os.listdir(hdir)):
        for sp in ("train.csv", "val.csv"):
            csvs.append(os.path.join(hdir, h, sp))
    tdir = os.path.join(fed_dir, "tests")
    for t in sorted(os.listdir(tdir)):
        csvs.append(os.path.join(tdir, t))
    for path in csvs:
        df = pd.read_csv(path, dtype={"nih_patient_id": str})
        tag = os.path.relpath(path, fed_dir)
        for nid in df["nih_patient_id"].astype(str):
            if nid in seen and seen[nid] != tag:
                raise AssertionError(
                    f"NIH patient {nid} in both {seen[nid]} and {tag} - NOT disjoint")
            seen[nid] = tag
    LOG.info(f"[verify] {fed_dir}: {len(seen)} NIH patients, all splits disjoint OK")


# ----------------------------------------------------------------------
# image-cue cutpoints (fit on dev pool, consistent with predict_proba)
# ----------------------------------------------------------------------
def fit_cutpoints_from_dev(dev_dir: str, n_ref: int = 300, seed: int = 0) -> None:
    import torch  # noqa
    from cue_bank import fit_image_cutpoints
    from hospital_dataset import CsvImageDataset, make_loader
    from model import denormalize_to_gray

    ref_csv = os.path.join(dev_dir, "tests", "test_balanced.csv")
    ds = CsvImageDataset(ref_csv)
    if len(ds) > n_ref:
        idx = np.sort(np.random.default_rng(seed).choice(len(ds), n_ref, replace=False))
        ds.df = ds.df.iloc[idx].reset_index(drop=True)
        ds.ids = ds.df["patientId"].astype(str).tolist()
        ds.targets = ds.df["Target"].astype(int).to_numpy()
        ds.cue_arrays = {k: v[idx] for k, v in ds.cue_arrays.items()}
    loader = make_loader(ds, batch_size=32, shuffle=False, num_workers=0)
    grays = []
    for imgs, _, _, _ in loader:
        grays.extend(list(denormalize_to_gray(imgs)))
    fit_image_cutpoints(grays)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def build_federation(cfg: SplitConfig, name: str = "main",
                     cells_only: bool = False) -> None:
    ensure_dirs()
    set_seed(cfg.seed)
    meta = pd.read_csv(config.PATIENT_METADATA_CSV, dtype={"patientId": str,
                                                           "nih_patient_id": str})
    meta = meta.dropna(subset=["Target"]).copy()
    meta["Target"] = meta["Target"].astype(int)
    LOG.info(f"loaded {len(meta)} patients from {config.PATIENT_METADATA_CSV}")

    # split NIH patients into dev vs main (disjoint)
    rng = np.random.default_rng(cfg.seed)
    nih_ids = meta["nih_patient_id"].astype(str).unique()
    rng.shuffle(nih_ids)
    n_dev = int(cfg.dev_fraction * len(nih_ids))
    dev_set = set(nih_ids[:n_dev])
    dev_pool = meta[meta["nih_patient_id"].astype(str).isin(dev_set)].reset_index(drop=True)
    main_pool = meta[~meta["nih_patient_id"].astype(str).isin(dev_set)].reset_index(drop=True)
    LOG.info(f"dev pool: {len(dev_pool)} imgs / {len(dev_set)} NIH patients | "
             f"main pool: {len(main_pool)} imgs")

    ctrl = control_cues_of(cfg.roster, cfg.balance_cue) or ["view", "sex"]
    LOG.info("MAIN pool cell availability:")
    report_pool_cells(main_pool, ctrl, cfg.prevalence, cfg.group_by_patient)
    if cells_only:
        LOG.info("--cells: diagnostic only, not building. Set sizes in config, "
                 "then run without --cells.")
        return

    main_dir = os.path.join(config.FEDERATIONS_DIR, name)
    dev_dir = os.path.join(config.FEDERATIONS_DIR, f"{name}_dev")

    LOG.info("=" * 60 + f"\nBUILDING DEV FEDERATION (scale={cfg.dev_scale}) -> {dev_dir}")
    build_one(dev_pool, dev_dir, cfg, np.random.default_rng(cfg.seed + 1),
              scale=cfg.dev_scale)
    verify_disjoint(dev_dir)

    LOG.info("=" * 60 + f"\nBUILDING MAIN FEDERATION -> {main_dir}")
    build_one(main_pool, main_dir, cfg, np.random.default_rng(cfg.seed + 2))
    verify_disjoint(main_dir)

    # cutpoints fit on dev (reference pool), used by every audit afterwards
    LOG.info("fitting image-cue cutpoints on dev reference pool ...")
    try:
        fit_cutpoints_from_dev(dev_dir, seed=cfg.seed)
    except Exception as e:
        LOG.warning(f"cutpoint fitting skipped ({e}); image cues unavailable "
                    "until cue_bank.fit_image_cutpoints is run.")
    LOG.info("federation build complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--coupling", type=float, default=None,
                    help="override coupling strength for all biased hospitals")
    ap.add_argument("--m", type=int, default=None,
                    help="number of biased hospitals to keep (0..2 for default roster)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cells", action="store_true",
                    help="print distinct-patient per-cell availability and exit")
    args = ap.parse_args()

    cfg = SplitConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.coupling is not None:
        for v in cfg.roster.values():
            if v["is_shortcut"]:
                v["coupling"] = args.coupling
    if args.m is not None:
        biased = [h for h, v in cfg.roster.items() if v["is_shortcut"]]
        for h in biased[args.m:]:
            cfg.roster[h] = {"cue": None, "coupling": 0.5, "is_shortcut": False}
    build_federation(cfg, name=args.name, cells_only=args.cells)


if __name__ == "__main__":
    main()

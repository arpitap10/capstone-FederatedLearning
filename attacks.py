"""
attacks.py - privacy attacks for the privacy-utility evaluation (plan 7.4, E8).

This is a pure *analysis* module: it quantifies how much private information
leaks through the artefacts the protocols release, under the threat model used
throughout the plan:

    A1  honest-but-curious SERVER  - sees per-client updates and released scores
    A2  honest-but-curious PEERS   - see each other's released reliability scores
    A4  malicious CLIENT           - lies about its own reported score to gain
                                     aggregation weight

Each function maps to one of those adversaries:

* ``membership_inference_loss_threshold`` / ``lira_lite``
      membership inference (A1): can an adversary with access to per-example
      losses / confidence scores tell whether a given patient was in training?
* ``attribute_inference_on_scores``
      attribute inference (A1/A2): can an observer of a client's released
      reliability scalars recover a sensitive property of that client's data
      (here its shortcut *coupling* strength)?
* ``score_inflation_attack`` / ``trimmed_mean_defense``
      a malicious client (A4) inflating its reported score, and a simple robust
      aggregation defence against it.

Everything takes plain numpy arrays / dicts and returns plain dicts of floats,
so the results drop straight into a report table.  No side effects on import.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from utils import get_logger

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LinearRegression
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - surfaced clearly at call time
    roc_auc_score = None
    LinearRegression = None
    spearmanr = None

import reliability  # for reliability.aggregation_weight

LOG = get_logger()


# ----------------------------------------------------------------------
def _attack_auroc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """AUROC of a membership score where HIGHER == more likely a member."""
    y = np.concatenate([np.ones(len(member_scores)),
                        np.zeros(len(nonmember_scores))])
    s = np.concatenate([np.asarray(member_scores, dtype=np.float64),
                        np.asarray(nonmember_scores, dtype=np.float64)])
    valid = ~np.isnan(s)
    if len(np.unique(y[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(y[valid], s[valid]))


# ======================================================================
# MEMBERSHIP INFERENCE  (A1 honest-but-curious server)
# ======================================================================
def membership_inference_loss_threshold(member_losses, nonmember_losses) -> dict:
    """Loss-threshold membership inference attack (Yeom et al. 2018).

    Threat model A1: the server (or anyone who can query the model on a known
    example) observes the per-example loss / negative-log-likelihood.  Training
    MEMBERS tend to have *lower* loss than held-out NON-MEMBERS, so a simple
    threshold on the loss separates them.

    We convert loss to a membership score ``-loss`` (higher == more likely a
    member) and report:
      * ``attack_auroc``   - ranking quality of the attack (0.5 == no leakage).
      * ``advantage``      - max over thresholds of (TPR - FPR), the
                             membership "advantage" of Yeom et al.
      * ``best_accuracy``  - balanced accuracy at the best threshold.
      * ``threshold``      - the loss threshold achieving ``advantage``
                             (predict member when loss <= threshold).

    Parameters
    ----------
    member_losses, nonmember_losses : 1-D array-like of per-example losses.
    """
    member = np.asarray(member_losses, dtype=np.float64)
    nonmember = np.asarray(nonmember_losses, dtype=np.float64)
    member = member[~np.isnan(member)]
    nonmember = nonmember[~np.isnan(nonmember)]
    if len(member) == 0 or len(nonmember) == 0:
        return {"attack_auroc": float("nan"), "advantage": float("nan"),
                "best_accuracy": float("nan"), "threshold": float("nan"),
                "n_member": int(len(member)), "n_nonmember": int(len(nonmember))}

    # Score = -loss so that "higher == member" for the AUROC convention.
    auroc = _attack_auroc(-member, -nonmember)

    # Sweep every candidate loss threshold; predict member when loss <= thr.
    cand = np.unique(np.concatenate([member, nonmember]))
    n_m, n_n = len(member), len(nonmember)
    best_adv, best_acc, best_thr = -np.inf, float("nan"), float("nan")
    for thr in cand:
        tpr = np.mean(member <= thr)              # members correctly flagged
        fpr = np.mean(nonmember <= thr)           # non-members wrongly flagged
        adv = tpr - fpr
        if adv > best_adv:
            # balanced accuracy of the member/non-member call at this threshold
            best_adv = adv
            best_acc = 0.5 * (tpr + (1.0 - fpr))
            best_thr = float(thr)
    return {"attack_auroc": auroc, "advantage": float(best_adv),
            "best_accuracy": float(best_acc), "threshold": best_thr,
            "n_member": int(n_m), "n_nonmember": int(n_n)}


def lira_lite(member_scores, nonmember_scores) -> dict:
    """Lightweight LiRA-style likelihood-ratio membership attack.

    Threat model A1.  This is a SIMPLIFIED, single-model approximation of the
    Likelihood Ratio Attack of Carlini et al. (2022), "Membership Inference
    Attacks From First Principles".  The full LiRA trains many shadow models to
    estimate per-example in/out Gaussians over a confidence statistic; here we
    make two simplifications:

      * a single pool of non-member scores stands in for the "out" distribution,
        fit with one global Gaussian N(mu_out, sigma_out^2) (no per-example
        calibration, no shadow models);
      * membership is scored by the one-sided tail position of a sample under
        that out-Gaussian, ``z = (x - mu_out) / sigma_out`` - a sample far in the
        upper tail of the non-member distribution is unlikely to be a
        non-member, hence likely a member (assuming members score higher, e.g.
        higher confidence / logit).

    This captures the *shape* of LiRA (likelihood ratio vs an out-distribution)
    while remaining a cheap, single-model diagnostic.  Reports ``attack_auroc``
    plus the fitted out-distribution parameters.

    Parameters
    ----------
    member_scores, nonmember_scores : 1-D array-like of a membership statistic
        (e.g. model confidence or logit) where larger == more member-like.
    """
    member = np.asarray(member_scores, dtype=np.float64)
    nonmember = np.asarray(nonmember_scores, dtype=np.float64)
    member = member[~np.isnan(member)]
    nonmember = nonmember[~np.isnan(nonmember)]
    if len(member) == 0 or len(nonmember) < 2:
        return {"attack_auroc": float("nan"), "mu_out": float("nan"),
                "sigma_out": float("nan"), "n_member": int(len(member)),
                "n_nonmember": int(len(nonmember))}

    mu_out = float(np.mean(nonmember))
    sigma_out = float(np.std(nonmember, ddof=1))
    if sigma_out <= 0:
        sigma_out = 1e-12

    # Membership score = standardized position in the out-distribution's upper
    # tail (monotone in the one-sided likelihood ratio under equal variances).
    z_member = (member - mu_out) / sigma_out
    z_nonmember = (nonmember - mu_out) / sigma_out
    auroc = _attack_auroc(z_member, z_nonmember)
    return {"attack_auroc": auroc, "mu_out": mu_out, "sigma_out": sigma_out,
            "n_member": int(len(member)), "n_nonmember": int(len(nonmember))}


# ======================================================================
# ATTRIBUTE INFERENCE  (A1 server / A2 peers observing released scores)
# ======================================================================
def attribute_inference_on_scores(released_scores_by_client: dict,
                                   true_coupling_by_client: dict) -> dict:
    """Attribute inference from released reliability scores.

    Threat model A1/A2: an observer collects each client's released reliability
    scalar(s) over the rounds and tries to recover a sensitive property of that
    client's data - here the shortcut *coupling* strength (a proxy for how biased
    the client's data is, which the protocol is supposed NOT to reveal directly).

    For each client we summarise its released scalars into a small feature vector
    (mean, std, min, max) and fit a linear regressor from those features to the
    true coupling.  The fit is evaluated in-sample: this is a *leakage* measure
    (how predictable coupling is from the released scores), not a generalisation
    estimate.  High Spearman / R^2 == the released scores leak the sensitive
    attribute.

    Parameters
    ----------
    released_scores_by_client : dict  client -> list/array of released scalars.
    true_coupling_by_client   : dict  client -> float coupling strength.

    Returns
    -------
    dict with ``spearman``, ``r2``, ``n_clients`` and the fitted ``coef``.
    """
    clients = [c for c in sorted(released_scores_by_client)
               if c in true_coupling_by_client]
    feats, y = [], []
    for c in clients:
        arr = np.asarray(released_scores_by_client[c], dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        feats.append([float(np.mean(arr)), float(np.std(arr)),
                      float(np.min(arr)), float(np.max(arr))])
        y.append(float(true_coupling_by_client[c]))

    n = len(y)
    if n < 2:
        return {"spearman": float("nan"), "r2": float("nan"),
                "n_clients": int(n), "coef": []}

    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if LinearRegression is not None:
        reg = LinearRegression().fit(X, y)
        pred = reg.predict(X)
        coef = list(map(float, reg.coef_))
    else:  # numpy least-squares fallback (design matrix with intercept)
        A = np.hstack([X, np.ones((n, 1))])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ beta
        coef = list(map(float, beta[:-1]))

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if spearmanr is not None and len(np.unique(pred)) > 1 and len(np.unique(y)) > 1:
        rho, _ = spearmanr(pred, y)
        rho = float(rho)
    else:
        rho = float("nan")

    return {"spearman": rho, "r2": float(r2), "n_clients": int(n), "coef": coef}


# ======================================================================
# SCORE-INFLATION ATTACK + DEFENCE  (A4 malicious client)
# ======================================================================
def score_inflation_attack(true_r: float, cfg) -> dict:
    """A malicious client inflates its reliability score to gain weight.

    Threat model A4.  The aggregation weight is
    ``w = n * max(r, r_min)^gamma`` (``reliability.aggregation_weight``).  A
    dishonest client reports ``r' = min(1.0, true_r + delta)`` for a grid of
    inflation amounts ``delta`` and we report, for each, the MULTIPLICATIVE gain
    in its weight relative to honest reporting (the shared ``n`` cancels, so we
    use ``n = 1``)::

        gain(delta) = max(r', r_min)^gamma / max(true_r, r_min)^gamma

    The ``r' <= 1`` clip caps the inflation: the largest achievable gain is at
    ``r' = 1``, giving ``max_gain_cap = 1 / max(true_r, r_min)^gamma``.  So the
    exponent ``gamma`` and the achievable score ceiling of 1.0 bound how much a
    single lie can buy - this is the knob the plan tunes for robustness.

    Parameters
    ----------
    true_r : float   the client's honest reliability score.
    cfg    : config.ReliabilityConfig  (supplies gamma, r_min).

    Returns
    -------
    dict with the ``deltas`` grid, per-delta ``gains``, the reported ``r_prime``
    values, and ``max_gain_cap``.
    """
    deltas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    honest_w = reliability.aggregation_weight(1, float(true_r), cfg)
    gains, r_primes = [], []
    for d in deltas:
        r_prime = min(1.0, float(true_r) + d)
        w = reliability.aggregation_weight(1, r_prime, cfg)
        r_primes.append(float(r_prime))
        gains.append(float(w / honest_w) if honest_w > 0 else float("inf"))
    # The clip at r' = 1 caps the inflation.
    w_max = reliability.aggregation_weight(1, 1.0, cfg)
    max_gain_cap = float(w_max / honest_w) if honest_w > 0 else float("inf")
    return {"true_r": float(true_r), "gamma": float(cfg.gamma),
            "r_min": float(cfg.r_min), "deltas": [float(x) for x in deltas],
            "r_prime": r_primes, "gains": gains, "max_gain_cap": max_gain_cap}


def trimmed_mean_defense(values, trim_frac: float = 0.1) -> float:
    """Trimmed-mean robust aggregation of a 1-D array of reported scalars.

    A simple defence (threat model A4) against a few clients inflating (or
    deflating) their reported scalars: sort the values, drop the top and bottom
    ``trim_frac`` fraction, and average the rest.  With ``trim_frac`` chosen above
    the expected fraction of malicious clients, a single inflated report is
    discarded before it can move the aggregate.

    ``trim_frac`` is clamped to [0, 0.5).  If trimming would remove every value,
    the plain mean (or median for an empty remainder) is returned.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan")
    frac = min(max(float(trim_frac), 0.0), 0.4999)
    v = np.sort(v)
    k = int(np.floor(frac * v.size))
    if 2 * k >= v.size:  # would trim everything -> fall back to median
        return float(np.median(v))
    trimmed = v[k: v.size - k] if k > 0 else v
    return float(np.mean(trimmed))

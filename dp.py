"""
dp.py - differential-privacy primitives for the federated protocols (plan 7.4).

This module supplies the *noise* half of the DP pipeline; the *clipping* half
already lives in ``model.clip_state`` (L2 clipping of the per-client update
Delta_i) and ``utils.clip01`` (range clipping of a released scalar score).  The
order is always CLIP-THEN-NOISE:

    1. bound the sensitivity of the quantity being released
         - an update Delta_i is clipped to L2 norm <= C   (model.clip_state)
         - a reliability score lives in [0, 1], sensitivity 1 by construction
    2. add calibrated Gaussian noise to the (already bounded) quantity
         - scalar score : noisy_clip01   (sensitivity 1, std = sigma)
         - update sum   : add_state_noise (sensitivity C, std = sigma*C/n)

Mechanisms provided
-------------------
* Gaussian mechanism on a scalar score      -> ``noisy_clip01``
* Gaussian mechanism on an aggregated update -> ``add_state_noise``
* Report-Noisy-Max over a dict of scores     -> ``report_noisy_max``
  (release only the argmax "cue" cue, not every score, to limit the composition
  cost of releasing many correlated scalars)
* a Renyi-DP accountant for the (subsampled) Gaussian mechanism -> ``RDPAccountant``

SIMULATION NOTE
---------------
This is a *simulation* of the privacy pipeline for the research harness, not a
production deployment.  Secure aggregation is modelled elsewhere
(``protocols.secure_aggregate``) as plain summation of the clipped updates; here
we simply add the aggregate Gaussian noise to the summed/averaged global state,
which is exactly the distributed-Gaussian-mechanism view (the sum of each
client's share of the noise equals one draw of the server-side noise).  No real
cryptography, no real client isolation - the point is to measure the
privacy-utility trade-off, not to be a secure MPC implementation.

All randomness flows through a ``numpy.random.Generator`` (``rng``) passed in by
the caller, so an entire federated run is reproducible from a single seed.  The
accountant is numpy-only and carries no external DP-library dependency.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import model  # for model._is_float and model.clone_state
from utils import clip01


# ======================================================================
# GAUSSIAN MECHANISM - scalar score
# ======================================================================
def noisy_clip01(value: float, sigma: float, rng) -> float:
    """Gaussian mechanism on a scalar score with sensitivity 1.

    A reliability score lives in [0, 1], so changing one patient's contribution
    moves it by at most 1 (a conservative bound; see ``sensitivity_note`` for the
    tighter ~1/min(n_pos, n_neg) argument).  We therefore add N(0, sigma^2) and
    clip the result back to [0, 1].

    The clip is applied AFTER the noise (post-processing a DP output is free, so
    clipping the noised value does not spend any additional privacy budget).

    Parameters
    ----------
    value : float   the clean score in [0, 1].
    sigma : float   noise multiplier (== noise std, because sensitivity is 1).
                    sigma <= 0 is treated as "no noise".
    rng   : numpy.random.Generator
    """
    if sigma is None or sigma <= 0:
        return clip01(float(value))
    noised = float(value) + float(rng.normal(loc=0.0, scale=sigma))
    return clip01(noised)


# ======================================================================
# GAUSSIAN MECHANISM - aggregated model update
# ======================================================================
def add_state_noise(state, sigma: float, C: float, n_clients: int, rng):
    """Gaussian mechanism on an aggregated (post-summation) global state.

    Each client update was clipped to L2 norm <= C, so the sensitivity of the
    *sum* of ``n_clients`` updates to one client's data is C.  For the
    FedAvg-style average (sum / n_clients) the per-coordinate noise std that
    realises a noise multiplier ``sigma`` is::

        std = sigma * C / n_clients

    i.i.d. Gaussian noise with that std is added to every FLOAT coordinate of
    the state; integer buffers (e.g. BatchNorm ``num_batches_tracked``) are left
    untouched - they are not model parameters and carry no gradient signal.

    A NEW state is returned (the input is cloned first, matching the no-mutation
    convention of ``model.state_*``).  Noise is drawn via ``rng.normal`` and
    moved into a torch tensor so a run is bit-reproducible from the seed.

    Parameters
    ----------
    state     : OrderedDict[str, torch.Tensor]  the aggregated global state.
    sigma     : float   noise multiplier; sigma <= 0 returns a plain clone.
    C         : float   the L2 clip bound used on each client update.
    n_clients : int     number of clients summed into the aggregate.
    rng       : numpy.random.Generator
    """
    out = model.clone_state(state)
    if sigma is None or sigma <= 0 or C is None or C <= 0:
        return out
    n = max(int(n_clients), 1)
    std = float(sigma) * float(C) / float(n)
    for k, v in out.items():
        if not model._is_float(v):
            continue  # skip int buffers
        # Draw numpy noise (reproducible from rng), match dtype/shape, add.
        noise_np = rng.normal(loc=0.0, scale=std, size=tuple(v.shape))
        noise = torch.from_numpy(noise_np).to(dtype=v.dtype, device=v.device)
        out[k] = v + noise
    return out


# ======================================================================
# REPORT-NOISY-MAX
# ======================================================================
def report_noisy_max(values: Dict[str, float], sigma: float, rng) -> str:
    """Report-Noisy-Max: release only the argmax key, not every score.

    Each score (sensitivity 1, in [0, 1]) is perturbed with independent
    N(0, sigma^2) noise and the key with the largest noised value is returned.
    Releasing only the argmax (which cue is most leaked) rather than the whole
    vector keeps the composition cost of the release at that of a single query,
    which is the standard argument for the Report-Noisy-Max / exponential-style
    selection mechanism.

    Parameters
    ----------
    values : dict  name -> score in [0, 1]; must be non-empty.
    sigma  : float noise multiplier; sigma <= 0 returns the clean argmax.
    rng    : numpy.random.Generator

    Returns
    -------
    str : the selected key.
    """
    if not values:
        raise ValueError("report_noisy_max requires a non-empty `values` dict")
    best_key, best_val = None, -math.inf
    for key, val in values.items():
        v = float(val)
        if sigma is not None and sigma > 0:
            v = v + float(rng.normal(loc=0.0, scale=sigma))
        if v > best_val:
            best_val, best_key = v, key
    return best_key


# ======================================================================
# RENYI-DP ACCOUNTANT  (subsampled Gaussian mechanism)
# ======================================================================
# Default order grid.  Fractional orders are included because they give the
# tightest (eps, delta) conversion for the non-subsampled Gaussian; integer
# orders are required for the subsampled binomial bound (see _rdp_at_order).
_DEFAULT_ORDERS: List[float] = [
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0,
    10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 48.0, 64.0,
]


class RDPAccountant:
    """Renyi-DP accountant for the (optionally subsampled) Gaussian mechanism.

    The accountant tracks, for a grid of Renyi orders ``alpha``, the accumulated
    RDP ``epsilon(alpha)`` across a sequence of Gaussian-mechanism applications.
    RDP composes by simple addition at each order, which is what makes it the
    convenient accountant for the many-round federated setting.

    Formulas
    --------
    Non-subsampled Gaussian mechanism (sample_rate == 1), Mironov (2017),
    "Renyi Differential Privacy", Prop. 7::

        rdp(alpha) = alpha / (2 * sigma^2)                     (for all alpha > 1)

    Subsampled Gaussian mechanism under *Poisson* subsampling with rate q < 1,
    integer order alpha >= 2.  We use the binomial upper bound of Mironov,
    Talwar & Zhang (2019), "Renyi Differential Privacy of the Sampled Gaussian
    Mechanism", which bounds the Renyi divergence of the sampled mechanism by::

        rdp(alpha) <= (1/(alpha-1)) *
            log( sum_{k=0}^{alpha} C(alpha,k) (1-q)^{alpha-k} q^k
                 * exp( (k^2 - k) / (2 sigma^2) ) )

    (This is an upper bound and is only defined here for integer alpha; for
    non-integer orders under subsampling we set rdp = +inf so those orders are
    simply ignored by the min in ``get_epsilon``.)

    RDP -> (eps, delta) conversion, Mironov (2017), Prop. 3::

        eps(delta) = min_alpha [ rdp(alpha) - log(delta) / (alpha - 1) ]

    (A tighter conversion exists - Balle et al. 2020, Canonne et al. 2020,
    ``eps = rdp + log((alpha-1)/alpha) - (log(delta) + log(alpha))/(alpha-1)`` -
    but the classic bound above is standard and sufficient for reporting.)
    """

    def __init__(self, orders: Optional[Sequence[float]] = None):
        self.orders = np.asarray(list(orders) if orders is not None
                                 else _DEFAULT_ORDERS, dtype=np.float64)
        # accumulated RDP at each order (same length as self.orders)
        self._rdp = np.zeros_like(self.orders)
        self.steps = 0

    # -- single-order RDP of one application -------------------------------
    @staticmethod
    def _rdp_at_order(alpha: float, sigma: float, q: float) -> float:
        """RDP of ONE Gaussian-mechanism application at Renyi order ``alpha``."""
        if sigma <= 0:
            return math.inf
        if alpha <= 1:
            # Renyi divergence undefined/limiting at order 1; skip this order.
            return math.inf
        if q >= 1.0:
            # Non-subsampled closed form.
            return alpha / (2.0 * sigma * sigma)
        if q <= 0.0:
            return 0.0
        # Subsampled: binomial bound requires an integer order.
        if not float(alpha).is_integer():
            return math.inf
        a = int(alpha)
        # Build log of each binomial term and log-sum-exp for stability.
        log_terms = []
        for k in range(a + 1):
            log_coef = math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)
            log_term = (log_coef
                        + (a - k) * math.log1p(-q)
                        + k * math.log(q)
                        + (k * k - k) / (2.0 * sigma * sigma))
            log_terms.append(log_term)
        m = max(log_terms)
        log_sum = m + math.log(sum(math.exp(t - m) for t in log_terms))
        return log_sum / (a - 1.0)

    # -- public API --------------------------------------------------------
    def step(self, sigma: float, sample_rate: float = 1.0) -> None:
        """Accumulate one Gaussian-mechanism application.

        Parameters
        ----------
        sigma       : float  noise multiplier of this application.
        sample_rate : float  Poisson subsampling rate q in (0, 1]; default 1.0
                             (every client/record participates each round).
        """
        add = np.array([self._rdp_at_order(float(a), float(sigma),
                                           float(sample_rate))
                        for a in self.orders], dtype=np.float64)
        self._rdp = self._rdp + add
        self.steps += 1

    def get_epsilon(self, delta: float) -> float:
        """Convert the accumulated RDP to an (eps, delta)-DP guarantee."""
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if self.steps == 0:
            return 0.0
        eps = np.full_like(self.orders, np.inf)
        mask = self.orders > 1.0
        eps[mask] = (self._rdp[mask]
                     - math.log(delta) / (self.orders[mask] - 1.0))
        best = float(np.min(eps))
        return best

    def get_privacy_spent(self, delta: float) -> Dict[str, float]:
        """Return eps and the order that achieves it (handy for logging)."""
        if self.steps == 0:
            return {"epsilon": 0.0, "best_order": float("nan"), "delta": delta}
        eps = np.full_like(self.orders, np.inf)
        mask = self.orders > 1.0
        eps[mask] = (self._rdp[mask]
                     - math.log(delta) / (self.orders[mask] - 1.0))
        i = int(np.argmin(eps))
        return {"epsilon": float(eps[i]), "best_order": float(self.orders[i]),
                "delta": float(delta), "steps": int(self.steps)}


# ======================================================================
# SENSITIVITY DOCUMENTATION
# ======================================================================
def sensitivity_note() -> str:
    """Short note documenting the AUROC-score sensitivity argument (plan 7.4).

    The conservative bound used by ``noisy_clip01`` is sensitivity 1 (a score in
    [0,1] can move by at most 1).  The tighter data-dependent bound referenced in
    the plan comes from the fact that adding or removing a single patient changes
    an AUROC estimate by at most ~1/min(n_pos, n_neg): AUROC is a normalised
    rank statistic over the n_pos * n_neg positive/negative pairs, and one
    patient participates in at most max(n_pos, n_neg) of those pairs, so the
    per-patient change is O(1 / min(n_pos, n_neg)).  With the plan's
    ``min_cell`` floor on each cue x label cell this gives a much smaller
    effective sensitivity than 1, which could be used to scale down the noise.
    """
    return (
        "AUROC score sensitivity: a released score lies in [0,1] (sensitivity 1, "
        "used by noisy_clip01). Tighter bound: adding/removing one patient shifts "
        "an AUROC estimate by at most ~1/min(n_pos, n_neg), since AUROC is a "
        "normalised rank statistic over n_pos*n_neg pairs and one patient enters "
        "at most max(n_pos,n_neg) of them. The min_cell floor bounds the smallest "
        "class count, hence the effective per-patient sensitivity."
    )

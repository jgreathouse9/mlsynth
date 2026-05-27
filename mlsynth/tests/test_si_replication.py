"""Self-contained replication of Agarwal-Shah-Shen (2026) with mlsynth's SI.

This pins ``mlsynth`` against the paper's *published target numbers* using only
mlsynth code plus public data -- no dependency on the authors' replication
package. The one-time machine-precision check against their actual code lives in
the docs ("Replication against the authors' code (Path A)"); here we lock in the
results so they cannot silently regress.

* Path A (empirical): the Proposition 99 case study on the vendored public
  pack-sales panel (``basedata/prop99_packsales.csv``).
* Path B (Monte Carlo): the consistency (Sec 5.1) and inference-coverage
  (Sec 5.2) studies, using mlsynth's own DGP reimplementations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mlsynth import SI
from mlsynth.utils.clustersc_helpers.pcr.hsvt import hsvt
from mlsynth.utils.si_helpers.estimation import (
    bias_corrected_fit,
    resolve_rank,
    select_omega,
    si_pcr_weights,
    variance_estimation,
)
from mlsynth.utils.si_helpers.simulation import (
    generate_low_rank_matrix,
    generate_low_rank_matrices,
)

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "prop99_packsales.csv"

TAX = ["Alaska", "Hawaii", "Maryland", "Michigan", "New Jersey", "New York", "Washington"]
PROGRAM = ["Arizona", "Massachusetts", "Oregon", "Florida"]   # California is also a program state
FIT_YEARS = list(range(1970, 1989))
INIT = {"control": 1989, "taxes": 1999, "program": 1998}
PRED = {iv: list(range(INIT[iv], INIT[iv] + 4)) for iv in INIT}   # T1 = 4


@pytest.fixture(scope="module")
def panel():
    if not _DATA.exists():
        pytest.skip(f"vendored Prop 99 data not found at {_DATA}")
    return pd.read_csv(_DATA)


# ----------------------------------------------------------------------
# Path A: the Proposition 99 case study (Section 6)
# ----------------------------------------------------------------------

class TestPathA_CaseStudy:
    def test_california_counterfactual_table(self, panel):
        """California's 1999-2002 counterfactual under each intervention."""
        d = panel[(panel.year <= 1988) | ((panel.year >= 1999) & (panel.year <= 2002))].copy()
        treated = set(TAX) | set(PROGRAM) | {"California"}
        d["control"] = (~d.state.isin(treated)).astype(int)
        d["taxes"] = d.state.isin(TAX).astype(int)
        d["program"] = d.state.isin(PROGRAM + ["California"]).astype(int)
        d["Prop99"] = ((d.state == "California") & (d.year >= 1999)).astype(int)

        res = SI({
            "df": d, "outcome": "cigsale", "unitid": "state", "time": "year",
            "treat": "Prop99", "inters": ["control", "taxes", "program"],
            "interval": "prediction", "display_graphs": False,
        }).fit()

        # paper Section 6.2.1: k = 5 (control), k = 1 (taxes, program)
        assert res.arms["control"].selected_rank == 5
        assert res.arms["taxes"].selected_rank == 1
        assert res.arms["program"].selected_rank == 1
        # published counterfactual means
        assert res.arms["control"].cf_mean == pytest.approx(75.78, abs=0.3)
        assert res.arms["taxes"].cf_mean == pytest.approx(57.53, abs=0.3)
        assert res.arms["program"].cf_mean == pytest.approx(59.12, abs=0.3)
        # control counterfactual sits well above tax / program
        assert res.arms["control"].cf_mean > res.arms["taxes"].cf_mean + 10

    def test_validation_coverage(self, panel):
        """Each state predicted under its own intervention; PI covers observed."""
        states = panel.state.unique()
        ctrl = [s for s in states if s not in set(TAX) | set(PROGRAM) | {"California"}]
        iv_states = {"control": ctrl, "taxes": TAX, "program": PROGRAM + ["California"]}
        iv_map = {**{s: "control" for s in ctrl}, **{s: "taxes" for s in TAX},
                  **{s: "program" for s in PROGRAM + ["California"]}}
        wide = panel.pivot_table(index="state", columns="year", values="cigsale")

        covered = {iv: [] for iv in INIT}
        for state in states:
            iv = iv_map[state]
            donors = [s for s in iv_states[iv] if s != state]
            ypt = wide.loc[state, FIT_YEARS].to_numpy()
            ypd = wide.loc[donors, FIT_YEARS].to_numpy().T
            ypost = wide.loc[donors, PRED[iv]].to_numpy().T
            obs = float(wide.loc[state, PRED[iv]].mean())
            T1 = len(PRED[iv])

            k = resolve_rank(ypd, "donoho")
            Mh, U, _, Vt = hsvt(ypd, k)
            sigma = variance_estimation(U, Vt.T, ypt, ypost)[0]
            om = select_omega(ypd, k)
            w = np.linalg.pinv(Mh[:, om]) @ ypt
            theta = float(np.mean(ypost[:, om] @ w))
            half = norm.ppf(0.975) * sigma * np.sqrt(1 + np.linalg.norm(w) ** 2) / np.sqrt(T1)
            covered[iv].append(theta - half <= obs <= theta + half)

        # published validation coverage: 26/38, 6/7, 3/5
        assert sum(covered["control"]) == 26 and len(covered["control"]) == 38
        assert sum(covered["taxes"]) == 6 and len(covered["taxes"]) == 7
        assert sum(covered["program"]) == 3 and len(covered["program"]) == 5


# ----------------------------------------------------------------------
# Path B: Monte Carlo studies (Sections 5.1 / 5.2)
# ----------------------------------------------------------------------

class TestPathB_Consistency:
    def test_consistent_only_when_rank_condition_holds(self):
        """SI-PCR error vanishes with T0 under A8, but not when A8 fails."""
        rng = np.random.default_rng(21248)
        noise = 1.0
        err = {"holds": {}, "fails": {}}
        for T0 in (40, 200):
            N, T1 = T0, 1
            r = int(np.power(N, 1 / 3)); r_pre = max(int(r / 2), 1)
            eh = ef = 0.0; cnt = 0
            for _ in range(5):
                A_in, A_out = generate_low_rank_matrices(N, T0, T1, r, r_pre, rng=rng)
                th_in = A_in[T0:, -1].mean(); th_out = A_out[T0:, -1].mean()
                for _ in range(40):
                    E = rng.normal(0, noise, (T0 + T1, N))
                    Yi, Yo = A_in + E, A_out + E
                    donor_pre = Yi[:T0, :-1]; target_pre = Yi[:T0, -1]
                    post_in = Yi[T0:, :-1]; post_out = Yo[T0:, :-1]
                    w = si_pcr_weights(donor_pre, target_pre, resolve_rank(donor_pre, "donoho"))
                    eh += abs(np.mean(post_in @ w) - th_in)
                    ef += abs(np.mean(post_out @ w) - th_out)
                    cnt += 1
            err["holds"][T0] = eh / cnt
            err["fails"][T0] = ef / cnt

        # at each T0, the rank-condition-holds error is far smaller
        for T0 in (40, 200):
            assert err["holds"][T0] < err["fails"][T0]
        # and it shrinks with T0 (consistency); the failing case does not
        assert err["holds"][200] < err["holds"][40]


class TestPathB_Inference:
    def _coverage(self, T0, rng):
        scale = 3.0; noise = 0.2 * scale; r = 8; alpha = 0.05
        N = int(np.power(T0, 2 / 3)); T1 = int(np.power(T0, 1 / 3)); T = T0 + T1
        cov = cnt = 0
        for _ in range(6):
            A = generate_low_rank_matrix(N, T0, T1, r, rng=rng)
            mx = np.max(np.abs(A))
            if mx > 1:  # the authors' [-1, 1] normalisation
                A = A / mx
            A = A * scale
            theta = A[T0:, -1].mean()
            for _ in range(50):
                Y = A + rng.normal(0, noise, (T, N))
                donor_pre = Y[:T0, :-1]; target_pre = Y[:T0, -1]
                donor_post = Y[T0:, :-1]
                om, w, _ = bias_corrected_fit(donor_pre, target_pre, r)
                _, U, _, Vt = hsvt(donor_pre, r)
                sigma = variance_estimation(U, Vt.T, target_pre, donor_post)[0]
                theta_hat = float(np.mean(donor_post[:, om] @ w))
                half = norm.ppf(1 - alpha / 2) * sigma * np.linalg.norm(w) / np.sqrt(T1)
                cov += theta_hat - half <= theta <= theta_hat + half
                cnt += 1
        return cov / cnt

    def test_ci_coverage_approaches_nominal(self):
        """Coverage rises toward the nominal 95% as T0 grows (Theorem 2)."""
        rng = np.random.default_rng(21248)
        cov_small = self._coverage(80, rng)
        cov_large = self._coverage(600, rng)
        # asymptotic normality: coverage improves with T0 and nears nominal
        assert cov_large >= cov_small
        assert 0.90 <= cov_large <= 0.99
        assert cov_small >= 0.80          # already reasonable at small T0

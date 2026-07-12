"""Clean-room staggered-adoption synthetic-control uncertainty engine.

A from-scratch (MIT) reproduction of the multiple-treated-unit prediction
intervals of Cattaneo, Feng, Palomba & Titiunik (2025), Section 4. No part of
the GPL ``scpi`` package is imported: data preparation, simplex weight
estimation, the in-sample conic prediction intervals (sub-Gaussian / location-
scale / quantile out-of-sample, plus the joint/uniform band) are all
reimplemented from the published methodology. Weight estimation uses ``cvxpy``
(CLARABEL); the in-sample conic prediction-interval program uses ``scpi``'s exact
second-order-cone construction solved directly with ``ecos`` (a cvxpy
reformulation diverges on the near-null directions the per-cell predictand
exercises). The engine has been validated numerically against ``scpi`` to solver
tolerance on the canonical Germany reunification panel -- point estimates and the
unit / unit-time / time prediction intervals, with and without covariates, agree
to a few decimals.

The cross-unit (effect="time", TSUA) in-sample interval carries a subtle scaling
choice. ``scpi`` divides the predictand matrix ``P`` by the number of treated
units ``iota`` in ``scdataMulti`` (correct -- it forms the average), and then
divides the simulated in-sample draws by ``iota`` a *second* time in
``scpi_in_diag`` -- a ``1 / iota**2`` shrinkage of the in-sample term, while the
point estimate and the out-of-sample term use the statistically correct single
``1 / iota``. This was verified against ``scpi`` at machine precision: removing
only the second division rescales ``scpi``'s in-sample width by exactly ``iota``
at every event time. ``tsua_double_divide`` selects which scaling to use:
``True`` reproduces ``scpi``'s published numbers bit-for-bit; ``False`` (the
library default) uses the statistically correct ``1 / iota``.

This module is self-contained and imported by the staggered ``VanillaSC`` path
for its causal-predictand prediction intervals; the surrounding library reuses
its own ``dataprep`` / point estimates for everything else.
"""
import numpy as np
import pandas as pd
from math import sqrt, log, ceil
import scipy.linalg
from scipy import sparse
from scipy.linalg import eigh, sqrtm
import cvxpy as cp
import ecos
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Per-unit data preparation  (mirrors scdata)
# ---------------------------------------------------------------------------
def complete_cases(x):
    return np.all(np.invert(np.isnan(np.asarray(x, dtype=float))), axis=1)


def scdata_unit(data, outcome_var, period_pre, period_post, unit_tr, unit_co,
                features, cov_adj, constant, cointegrated_data):
    """Build A, B, C, P, Y_donors for a single treated unit."""
    features = sorted(features)
    M = len(features)
    out_in_features = outcome_var in features

    d = data.copy()
    d['__ID'] = d['country'].astype(str)
    d['__time'] = d['year']
    d['treated_unit'] = unit_tr
    d = d.set_index(['treated_unit', '__ID', '__time'], drop=False)

    period_pre = np.sort(np.array(period_pre))
    period_post = np.sort(np.array(period_post))

    # balanced panel for the selected units (fill missing unit-time with NaN)
    sub = d[d['__ID'].isin([unit_tr] + list(unit_co))]
    ids = sorted(sub['__ID'].unique().tolist())
    times = sorted(sub['__time'].unique().tolist())
    full_idx = pd.MultiIndex.from_product([[unit_tr], ids, times],
                                          names=['treated_unit', '__ID', '__time'])
    data_bal = sub.reindex(full_idx)
    data_bal['__ID'] = data_bal.index.get_level_values('__ID')
    data_bal['__time'] = data_bal.index.get_level_values('__time')
    data_bal['treated_unit'] = unit_tr

    rows_tr_pre = data_bal.loc[(unit_tr, [unit_tr], period_pre), ]
    rows_tr_post = data_bal.loc[(unit_tr, [unit_tr], period_post), ]
    rows_co_pre = data_bal.loc[(unit_tr, unit_co, period_pre), ]
    rows_co_post = data_bal.loc[(unit_tr, unit_co, period_post), ]

    Y_pre = rows_tr_pre[[outcome_var]].reset_index(level='__ID', drop=True)
    Y_pre.columns = [unit_tr]

    # A: treated features stacked
    A_df = pd.melt(rows_tr_pre, id_vars=['treated_unit', '__ID', '__time'],
                   value_vars=features, var_name='feature', value_name=unit_tr)
    A = A_df.loc[:, [unit_tr]]

    # B: donor features, wide by donor
    B_df = pd.melt(rows_co_pre, id_vars=['treated_unit', '__ID', '__time'],
                   value_vars=features, var_name='feature', value_name='value')
    B = B_df.pivot(index=['treated_unit', 'feature', '__time'],
                   columns='__ID', values='value')
    donor_order = B.columns.values.tolist()

    sel = rows_co_pre[[outcome_var, 'treated_unit', '__time', '__ID']]
    Y_donors = sel.pivot(index=['treated_unit', '__time'], columns='__ID',
                         values=outcome_var)[donor_order]

    # C: covariate-adjustment block
    C = pd.DataFrame(None)
    C_names = []
    if cov_adj is not None:
        if not isinstance(cov_adj[0], list):
            covs = [c for c in cov_adj if c not in ('constant', 'trend')]
            Cc = pd.DataFrame(None)
            if 'constant' in cov_adj:
                Cc['constant'] = np.ones(len(rows_tr_pre))
            if 'trend' in cov_adj:
                Cc['trend'] = period_pre - period_pre[0] + 1
            rows_C = data_bal.loc[(unit_tr, unit_co[0], period_pre), covs].reset_index()
            for c in covs:
                Cc[c] = rows_C[c].values
            for num in range(1, M + 1):
                for cov in Cc.columns:
                    C_names.append(f"{num}_{cov}")
            C_arr = np.kron(np.identity(M), np.array(Cc))
            C = pd.DataFrame(C_arr, columns=C_names)
        else:
            for m in range(M):
                covs_all = cov_adj[m]
                covs = [c for c in covs_all if c not in ('constant', 'trend')]
                Cm = pd.DataFrame(None)
                if 'constant' in covs_all:
                    Cm['constant'] = np.ones(len(rows_tr_pre))
                if 'trend' in covs_all:
                    Cm['trend'] = period_pre - period_pre[0] + 1
                rows_C = data_bal.loc[(unit_tr, unit_co[0], period_pre), covs].reset_index()
                for c in covs:
                    Cm[c] = rows_C[c].values
                if m == 0:
                    C = Cm
                else:
                    C = scipy.linalg.block_diag(np.array(C), np.array(Cm))
                for cov in cov_adj[m]:
                    C_names.append(f"{m + 1}_{cov}")
            C = pd.DataFrame(C, columns=C_names)

    if constant is True:
        glob = pd.DataFrame(np.ones(len(B)), columns=['0_constant'])
        if len(C.columns) == 0:
            C = glob
        else:
            C.insert(0, '0_constant', glob)
        C_names.insert(0, '0_constant')

    Y_post = rows_tr_post[[outcome_var]].reset_index(level='__ID', drop=True)
    Y_post.columns = [unit_tr]

    # P: prediction matrix
    P = rows_co_post[[outcome_var, 'treated_unit', '__ID', '__time']].pivot(
        index=['treated_unit', '__time'], columns='__ID', values=outcome_var)[donor_order]

    if out_in_features:
        if constant is True:
            P['0_constant'] = np.ones(len(rows_tr_post))
        if cov_adj is not None:
            for m in range(1, M + 1):
                if not isinstance(cov_adj[0], list):
                    covs_all = list(cov_adj)
                else:
                    covs_all = list(cov_adj[m - 1])
                if features[m - 1] == outcome_var:
                    if 'constant' in covs_all:
                        P[f"{m}_constant"] = np.ones(len(rows_tr_post))
                        covs_all.remove('constant')
                    if 'trend' in covs_all:
                        P[f"{m}_trend"] = period_post - period_pre[0] + 1
                        covs_all.remove('trend')
                    rows_P = data_bal.loc[(unit_tr, unit_co[0], period_post), covs_all].reset_index()
                    for c in covs_all:
                        P[f"{m}_{c}"] = rows_P[c].values
                else:
                    for c in covs_all:
                        P[f"{m}_{c}"] = np.zeros(len(P.index))

    T1 = len(period_post)

    # drop donors fully missing in pre-period
    empty = np.all(np.isnan(np.asarray(B, dtype=float)), axis=0)
    unit_co_eff = [co for co, e in zip(donor_order, empty) if not e]
    B = B[unit_co_eff]

    A.set_index(B.index, drop=True, inplace=True)
    if len(C.columns) > 0:
        C.set_index(B.index, drop=True, inplace=True)

    X = A.join([B, C], how='outer', sort=True)
    X_na = X.loc[complete_cases(X), ]
    A_na = X_na.loc[:, [unit_tr]]

    if len(unit_co_eff) < len(donor_order):
        Ccols = [c for c in P.columns.tolist() if c not in donor_order]
        donor_order = unit_co_eff
        Y_donors = Y_donors[donor_order]
        P = P[donor_order + Ccols]

    B_na = X_na[donor_order]
    C_na = X_na.loc[:, C_names] if len(C_names) > 0 else pd.DataFrame(None)

    from collections import Counter
    T0_features = Counter(A_na.index.get_level_values('feature'))

    J = len(unit_co_eff)
    KM = len(C.columns)

    A_na.columns = ['A']
    B_na.columns = unit_tr + '_' + B_na.columns
    if len(C_na.columns) > 0:
        C_na.columns = unit_tr + '_' + C_na.columns
    P.columns = unit_tr + '_' + P.columns
    Y_donors.columns = unit_tr + '_' + Y_donors.columns

    A_na.index.names = ['ID', 'feature', 'Time']
    B_na.index.names = ['ID', 'feature', 'Time']
    if len(C_na.columns) > 0:
        C_na.index.names = ['ID', 'feature', 'Time']
    P.index.names = ['ID', 'Time']
    Y_pre.index.names = ['ID', 'Time']
    Y_post.index.names = ['ID', 'Time']
    Y_donors.index.names = ['ID', 'Time']

    if cointegrated_data and any(t == 1 for t in T0_features.values()):
        cointegrated_data = False

    return dict(A=A_na, B=B_na, C=C_na, P=P, Y_pre=Y_pre, Y_post=Y_post,
                Y_donors=Y_donors, J=J, KM=KM, M=M, T0_features=dict(T0_features),
                T1=T1, out_in_features=out_in_features, donors=unit_co_eff,
                period_pre=period_pre, period_post=period_post,
                cointegrated_data=cointegrated_data, glob_cons=constant)


def scdata_multi(data, outcome_var, features, cov_adj, constant,
                 cointegrated_data, effect):
    """Stack per-unit blocks into the block-diagonal multi-unit design."""
    # treated units = those ever treated
    treated_units = sorted(data.loc[data['status'] == 1, 'country'].unique().tolist())
    treated_post = list(treated_units)

    treated_periods = (data.loc[data['status'] == 1, ['country', 'year']]
                       .groupby('country').min()['year'].to_dict())

    blocks = {}
    P_blocks = []
    Pd_blocks = []
    for tr in treated_units:
        T0_date = treated_periods[tr]
        donors = sorted(data.loc[(~data['country'].isin(treated_post)) &
                                 (data['year'] < T0_date), 'country'].unique().tolist())
        # not-yet-treated also enter as donors for unit-time / unit
        # (here both treated units are excluded throughout; donors are the 15 OECD)
        time_all = sorted(data.loc[data['country'].isin(donors + [tr]), 'year'].unique())
        period_pre = [t for t in time_all if t < T0_date]
        period_post = [t for t in time_all if t >= T0_date]

        feats = features[tr] if tr in features else list(features.values())[0]
        cadj = cov_adj[tr] if tr in cov_adj else list(cov_adj.values())[0]
        cst = constant[tr] if isinstance(constant, dict) else constant
        coi = cointegrated_data[tr] if isinstance(cointegrated_data, dict) else cointegrated_data

        out = scdata_unit(data, outcome_var, period_pre, period_post, tr, donors,
                          feats, cadj, cst, coi)
        blocks[tr] = out

        P_tr = out['P']
        P_diff = None
        if effect == "time":
            time = P_tr.index.get_level_values('Time').tolist()
            time = [t - min(time) + 1 for t in time]
            P_tr = pd.DataFrame(P_tr.values, index=time, columns=P_tr.columns)
        if effect == "unit":
            JJ = out['J']
            if out['cointegrated_data']:
                # out_in_features True here
                T0o = out['T0_features'][outcome_var]
                P_first = P_tr.iloc[[0], :JJ] - out['B'].iloc[[T0o - 1], :].values
                P_diff = P_tr.iloc[:, :JJ].diff()
                P_diff.iloc[0, :] = P_first
                P_diff = pd.concat([P_diff.iloc[:, :JJ], P_tr.iloc[1:, JJ:]], axis=1)
                aux = np.array([P_diff.mean(axis=0)])
                tcol = out['period_post'][ceil(out['T1'] / 2) - 1]
                idx = pd.MultiIndex.from_product([[tr], [tcol]], names=['ID', 'Time'])
                P_diff = pd.DataFrame(aux, index=idx, columns=P_diff.columns)
            aux = np.array([P_tr.mean(axis=0)])
            tcol = out['period_post'][ceil(out['T1'] / 2) - 1]
            idx = pd.MultiIndex.from_product([[tr], [tcol]], names=['ID', 'Time'])
            P_tr = pd.DataFrame(aux, index=idx, columns=P_tr.columns)
        P_blocks.append(P_tr)
        if P_diff is not None:
            Pd_blocks.append(P_diff)

    A = pd.concat([blocks[t]['A'] for t in treated_units], axis=0)
    B = pd.concat([blocks[t]['B'] for t in treated_units], axis=0)
    C = pd.concat([blocks[t]['C'] for t in treated_units], axis=0)
    if effect == "time":
        P = pd.concat(P_blocks, axis=1, join='inner')
    else:
        P = pd.concat(P_blocks, axis=0)
    Pd = pd.concat(Pd_blocks, axis=0) if Pd_blocks else None
    Y_donors = pd.concat([blocks[t]['Y_donors'] for t in treated_units], axis=0)

    B.set_index(A.index, inplace=True)
    if len(C.columns) > 0:
        C.set_index(A.index, inplace=True)
    else:
        C = pd.DataFrame(index=A.index)
    B = B.fillna(0)
    C = C.fillna(0)
    P = P.fillna(0)
    if Pd is not None:
        Pd = Pd.fillna(0)
    Y_donors = Y_donors.fillna(0)

    bcols = B.columns.tolist()
    ccols = C.columns.tolist()
    P = P[bcols + ccols]

    iota = len(treated_units)
    if effect == "time":
        P = P / iota
        T1min = min(blocks[t]['T1'] for t in treated_units)
        for t in treated_units:
            blocks[t]['T1'] = T1min

    return dict(A=A, B=B, C=C, P=P, P_diff=Pd, Y_donors=Y_donors,
                blocks=blocks, treated_units=treated_units, iota=iota,
                effect=effect, outcome_var=outcome_var)


# ---------------------------------------------------------------------------
# Weight estimation  (mirrors scest, V = separate -> per-unit problems)
# ---------------------------------------------------------------------------
def _mat2dict_rows(mat, treated_units):
    out = {}
    for tr in treated_units:
        out[tr] = mat.loc[pd.IndexSlice[tr, :, :]] if mat.index.nlevels == 3 \
            else mat.loc[pd.IndexSlice[tr, :]]
    return out


def _shrinkage_est(method, A, Z, J, KM):
    """scpi ``shrinkage_EST`` rule-of-thumb for one feature block (V = identity).

    Returns ``(Q, lambda)``. For ridge, ``lambda = sigma^2 (J + KM) / ||b||**2``
    and ``Q = ||b|| / (1 + lambda)`` from an unweighted OLS of ``A`` on ``Z``;
    when observations are scarce (``len(Z) <= #cols + 10``) scpi lasso-screens
    the columns first. Lasso returns ``Q = 1``.
    """
    Aarr = np.asarray(A, float).ravel()
    Zarr = np.asarray(Z, float)
    if method == "lasso":
        return 1.0, None

    def _rule(Zsub):
        n, k = Zsub.shape
        b, *_ = np.linalg.lstsq(Zsub, Aarr, rcond=None)
        resid = Aarr - Zsub @ b
        sig = float(resid @ resid) / max(n - k, 1)
        L2 = float(b @ b)
        lam = sig * (J + KM) / max(L2, 1e-12)
        return sqrt(L2) / (1.0 + lam), lam

    n, k = Zarr.shape
    if n > k + 10:
        return _rule(Zarr)
    # scarce-obs lasso screen (scpi), then refit on the kept columns
    z = cp.Variable(Zarr.shape[1])
    try:
        cp.Problem(cp.Minimize(cp.sum_squares(Aarr - Zarr @ z)),
                   [cp.norm1(z) <= 1.0]).solve(solver=cp.CLARABEL)
        coefs = np.abs(np.asarray(z.value).ravel())
    except Exception:  # pragma: no cover - solver fallback
        coefs = np.abs(np.linalg.lstsq(Zarr, Aarr, rcond=None)[0])
    keep = max(min(n - 10, k), 2)
    active = np.sort(np.argsort(coefs)[::-1][:keep])
    return _rule(Zarr[:, active])


def _w_constr_prep_unit(w_constr, A_df, B_df, J, KM):
    """Normalise a weight-constraint spec for one treated unit (scpi
    ``w_constr_prep``): fills ``p``/``dir``/``lb``/``Q``/``lambda`` and, for
    ridge / L1-L2, estimates the data-driven budget per feature (``max`` of the
    per-feature ``min`` Q, floored at 0.5)."""
    name = (w_constr if isinstance(w_constr, str)
            else dict(w_constr).get("name", "simplex"))
    spec = {} if isinstance(w_constr, str) else dict(w_constr)
    if name == "simplex":
        return {"name": "simplex", "p": "L1", "dir": "==", "lb": 0.0,
                "Q": float(spec.get("Q", 1.0)), "Q2": None, "lambda": None}
    if name == "ols":
        return {"name": "ols", "p": "no norm", "dir": None, "lb": -np.inf,
                "Q": None, "Q2": None, "lambda": None}
    if name == "lasso":
        return {"name": "lasso", "p": "L1", "dir": "<=", "lb": -np.inf,
                "Q": float(spec.get("Q", 1.0)), "Q2": None, "lambda": None}

    # ridge / L1-L2: data-driven Q per feature block on the donor design. A_df
    # and B_df are the treated pre-outcome and donor design, feature-indexed.
    has_feature = (isinstance(A_df, pd.DataFrame)
                   and "feature" in (A_df.index.names or []))
    feats = (A_df.index.get_level_values("feature").unique().tolist()
             if has_feature else [None])
    Qfeat, lam = [], None
    for f in feats:
        if f is not None:
            Af = np.asarray(A_df.xs(f, level="feature"), float).ravel()
            Bf = np.asarray(B_df.xs(f, level="feature"), float)
        else:
            Af = np.asarray(A_df, float).ravel()
            Bf = np.asarray(B_df, float)
        if Bf.shape[0] >= 5:
            try:
                Qe, lam = _shrinkage_est("ridge", Af, Bf, J, KM)
                Qfeat.append(Qe)
            except Exception:  # pragma: no cover
                pass
    if not Qfeat:
        Qe, lam = _shrinkage_est("ridge", np.asarray(A_df, float).ravel(),
                                 np.asarray(B_df, float), J, KM)
        Qfeat.append(Qe)
    Qest = max(float(np.nanmin(Qfeat)), 0.5)
    if name == "ridge":
        return {"name": "ridge", "p": "L2", "dir": "<=", "lb": -np.inf,
                "Q": float(spec.get("Q", Qest)), "Q2": None, "lambda": float(lam)}
    return {"name": "L1-L2", "p": "L1-L2", "dir": "==/<=", "lb": 0.0,
            "Q": 1.0, "Q2": float(spec.get("Q2", Qest)), "lambda": float(lam)}


def b_est(A, Z, J, KM, w_constr="simplex", B_donor=None):
    """Solve the synthetic-control weight problem for one treated unit under
    the scpi weight-constraint family. The constraint binds the donor block
    ``x[:J]``; covariate/constant coefficients ``x[J:]`` are unconstrained.

    Returns ``(x_value, wc)`` where ``wc`` is the normalised constraint dict
    (with the estimated ridge / lasso budget ``Q`` and penalty ``lambda``).
    """
    Zarr = np.asarray(Z, dtype=float)
    Aarr = np.asarray(A, dtype=float).reshape(-1, 1)
    Bd = B_donor if B_donor is not None else pd.DataFrame(Zarr[:, :J])
    wc = _w_constr_prep_unit(w_constr, A, Bd, J, KM)
    x = cp.Variable((J + KM, 1))
    obj = cp.Minimize(cp.sum_squares(Aarr - Zarr @ x))
    xd = x[0:J]
    name = wc["name"]
    if name == "simplex":
        cons = [cp.sum(xd) == wc["Q"], xd >= 0]
    elif name == "ols":
        cons = []
    elif name == "lasso":
        cons = [cp.norm1(xd) <= wc["Q"]]
    elif name == "ridge":
        cons = [cp.sum_squares(xd) <= wc["Q"] ** 2]
    else:  # L1-L2
        cons = [cp.sum(xd) == wc["Q"], xd >= 0,
                cp.sum_squares(xd) <= wc["Q2"] ** 2]
    prob = cp.Problem(obj, cons)
    for solver in (cp.CLARABEL, cp.ECOS, cp.SCS):
        try:
            prob.solve(solver=solver)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return x.value, wc
        except Exception:
            continue
    raise RuntimeError(f"estimation not converged: {prob.status}")


def b_est_simplex(A, Z, J, KM):
    """Solve the simplex synthetic-control QP for one treated unit (back-compat
    wrapper over :func:`b_est`)."""
    return b_est(A, Z, J, KM, "simplex")[0]


def scest(md, w_constr="simplex"):
    """Estimate weights per treated unit and assemble point predictions.

    ``w_constr`` is the scpi weight-constraint family (``"simplex"`` default, or
    ``"ols"`` / ``"lasso"`` / ``"ridge"`` / ``"L1-L2"``, or an explicit dict),
    applied per treated unit; the binding is on the donor block only.
    """
    treated_units = md['treated_units']
    A, B, C, P = md['A'], md['B'], md['C'], md['P']
    Z = pd.concat([B, C], axis=1)
    blocks = md['blocks']

    A_d = {tr: A.loc[pd.IndexSlice[tr, :, :]] for tr in treated_units}
    B_d = {}
    C_d = {}
    for tr in treated_units:
        Br = B.loc[pd.IndexSlice[tr, :, :]]
        Cr = C.loc[pd.IndexSlice[tr, :, :]]
        B_d[tr] = Br.loc[:, [c for c in Br.columns if c.split('_')[0] == tr]]
        C_d[tr] = Cr.loc[:, [c for c in Cr.columns if c.split('_')[0] == tr]]

    w_store = []
    r_store = []
    w_constr_inf = {}
    for tr in treated_units:
        J = blocks[tr]['J']
        KM = blocks[tr]['KM']
        Zi = pd.concat([B_d[tr], C_d[tr]], axis=1)
        res, w_constr_inf[tr] = b_est(A_d[tr], Zi, J, KM, w_constr, B_donor=B_d[tr])
        idx = pd.MultiIndex.from_product([[tr], blocks[tr]['donors']])
        w_store.append(pd.DataFrame(res[0:J], index=idx))
        if KM > 0:
            cnm = [c.split('_', 1)[1] for c in C_d[tr].columns.tolist()]
            idx = pd.MultiIndex.from_product([[tr], cnm])
            r_store.append(pd.DataFrame(res[J:], index=idx))

    w = pd.concat(w_store, axis=0)
    r = pd.concat(r_store, axis=0) if r_store else pd.DataFrame(None)
    b = pd.concat([w, r], axis=0)
    b.index.rename(['ID', 'donor'], inplace=True)
    w.index.rename(['ID', 'donor'], inplace=True)

    A_hat = Z.dot(np.asarray(b))
    A_hat.columns = A.columns
    res = A - A_hat

    fit_pre = A_hat.loc[pd.IndexSlice[:, md['outcome_var'], :]]
    fit_post = P.dot(np.asarray(b))
    fit_pre.columns = A.columns
    fit_post.columns = A.columns

    return dict(b=b, w=w, r=r, A_hat=A_hat, res=res, w_constr_inf=w_constr_inf,
                Y_pre_fit=fit_pre, Y_post_fit=fit_post, Z=Z, **md)


# ===========================================================================
# Uncertainty quantification  (mirrors scpi)
# ===========================================================================
def mat2dict(mat, treated_units, cols=True):
    X = mat.copy()
    out = {}
    if mat.shape[1] == 1:
        cols = False
    for tr in treated_units:
        Xr = X.loc[pd.IndexSlice[tr, :, :]] if X.index.nlevels == 3 \
            else X.loc[pd.IndexSlice[tr, :]]
        if 'ID' in (Xr.index.names or []):
            Xr = Xr.reset_index(level=0, drop=True)
        if cols:
            csel = [str(c).split('_')[0] == tr for c in Xr.columns.tolist()]
            Xr = Xr.loc[:, np.array(csel)]
        out[tr] = Xr
    return out


def regularize_w(rho_type, rho_max, res, B, T0_tot, J, KM, d0):
    ssr = (res - res.mean()) ** 2
    sigma_u = sqrt(float(np.asarray(ssr.mean()).reshape(-1)[0]))
    if rho_type == 'type-1':
        sigma_bj = min(B.std(axis=0))
        CC = sigma_u / sigma_bj
    else:  # type-2
        sigma_bj2 = min(B.var(axis=0))
        sigma_bj = max(B.std(axis=0))
        CC = sigma_u * sigma_bj / sigma_bj2
    d = J + KM
    rho = CC * sqrt(log(d) * d0 * log(T0_tot)) / sqrt(T0_tot)
    if rho_max is not None:
        rho = min(rho, rho_max)
    return rho


def regularize_check(w, index_w, rho, B):
    if sum(index_w) == 0:
        sel = w.rank(ascending=False) <= 1
        index_w = index_w | sel[0]
    return index_w


def regularize_check_lb(w, rho, rho_max, res, B, T0_tot, J, KM, d0):
    if rho < 0.001:
        rho = max(regularize_w("type-1", 0.2, res, B, T0_tot, J, KM, d0),
                  regularize_w("type-2", 0.2, res, B, T0_tot, J, KM, d0))
        if rho < 0.05:
            rho = rho_max
    return rho


def local_geom_simplex(rho, rho_max, res, B, T0_tot, J, KM, w):
    d0 = sum(abs(w.iloc[:, 0].values) >= 1e-6) + KM
    if isinstance(rho, str):
        rho = regularize_w(rho, rho_max, res, B, T0_tot, J, KM, d0)
        rho = regularize_check_lb(w, rho, rho_max, res, B, T0_tot, J, KM, d0)
    index_w = w[0] > rho
    index_w = regularize_check(w, index_w, rho, B)
    return rho, index_w.values


def localgeom2step_lb(w, rho_dict, treated_units, J_dict):
    """Lower bounds for the in-sample conic program (simplex, dir '==')."""
    lb = []
    jmin = 0
    warr = w.iloc[:, 0].values
    for tr in treated_units:
        jmax = jmin + J_dict[tr]
        wj = warr[jmin:jmax]
        active = 1 * (wj < rho_dict[tr])
        lb.extend((active * wj).tolist())
        jmin = jmax
    return lb


# ---------------------------------------------------------------------------
# Generalized weight-constraint family: local_geom / localgeom2step / df_EST
# (faithful port of scpi's funs.local_geom / localgeom2step / df_EST, covering
#  ols / simplex / lasso / ridge / L1-L2). The simplex-only helpers above are
#  kept so the validated simplex conic path is untouched.
# ---------------------------------------------------------------------------
def local_geom(w_constr, rho, rho_max, res, B, T0_tot, J, KM, w):
    """Regularise one treated unit's weights and localise the compatible set.

    Mirrors scpi's ``local_geom``: computes (or accepts) ``rho``, the active-donor
    set ``index_w`` (for the u/e design), and a preliminary norm budget
    ``Q_star`` / ``Q2_star``; mutates ``w_constr['Q']`` (and ``['Q2']`` for
    L1-L2) in place, as scpi does on its per-unit working copy. Returns
    ``(w_constr, index_w, rho, Q_star, Q2_star)``.
    """
    Q = w_constr["Q"]
    Q2_star = None
    name = w_constr["name"]
    p = w_constr.get("p")
    dire = w_constr.get("dir")
    d0 = int(np.sum(np.abs(w.iloc[:, 0].values) >= 1e-6)) + KM
    if isinstance(rho, str):
        rho = regularize_w(rho, rho_max, res, B, T0_tot, J, KM, d0)
        rho = regularize_check_lb(w, rho, rho_max, res, B, T0_tot, J, KM, d0)

    if name == "simplex" or (p == "L1" and dire == "=="):
        index_w = w[0] > rho
        index_w = regularize_check(w, index_w, rho, B)
        w_star = w.copy()
        w_star.loc[index_w.values] = 0
        Q_star = float(np.asarray(w_star).sum())
    elif name == "lasso" or (p == "L1" and dire == "<="):
        l1 = float(np.sum(np.abs(w)))
        Q_star = l1 if (Q - rho * sqrt(J) <= l1 <= Q) else Q
        index_w = np.abs(w[0]) > rho
        index_w = regularize_check(w, index_w, rho, B)
    elif name == "ridge" or p == "L2":
        l2 = float(sqrt(np.sum(np.asarray(w) ** 2)))
        Q_star = l2 if (Q - rho <= l2 <= Q) else Q
        index_w = pd.Series([True] * len(B.columns), index=w.index)
    elif name == "L1-L2":
        index_w = w[0] > rho
        index_w = regularize_check(w, index_w, rho, B)
        w_star = w.copy()
        w_star.loc[index_w.values] = 0
        Q_star = float(np.asarray(w_star).sum())
        l2 = float(sqrt(np.sum(np.asarray(w) ** 2)))
        Q2_star = l2 if (Q - rho <= l2 <= Q) else float(w_constr["Q2"])
        w_constr["Q2"] = Q2_star
    else:  # ols / no norm
        Q_star = Q
        index_w = pd.Series([True] * len(B.columns), index=w.index)

    w_constr["Q"] = Q_star
    return w_constr, np.asarray(index_w).astype(bool), rho, Q_star, Q2_star


def localgeom2step(w, rho_dict, w_constr, Q, treated_units, J_dict, rho_max=0.2):
    """Two-step refinement of the norm budget and per-donor lower bounds.

    Mirrors scpi's ``localgeom2step`` over all treated units: for inequality
    constraints (lasso / L1-L2) the norm budget is relaxed by ``rho`` when the
    realised norm binds; the lower bounds pin the small (regularised-away) donors
    for the lower-bounded constraints (simplex / L1-L2) and are ``-inf`` for the
    signed constraints (ols / lasso / ridge). Returns ``(Q_star_dict, lb_list)``.
    """
    from copy import deepcopy
    Q_star = deepcopy(dict(Q))
    lb = []
    warr = w.iloc[:, 0].values
    jmin = 0
    for tr in treated_units:
        jmax = jmin + J_dict[tr]
        wj = warr[jmin:jmax]
        p = w_constr[tr]["p"]
        dire = w_constr[tr]["dir"]
        lbc = w_constr[tr]["lb"]
        w_norm = None
        if p == "no norm":
            rhoj = rho_dict[tr]
        elif p == "L1":
            rhoj = rho_dict[tr]
            w_norm = float(np.sum(np.abs(wj)))
        else:  # "L1-L2" / "L2"
            w_norm = float(np.sum(wj ** 2))
            rhoj = min(2 * np.sqrt(w_norm) * rho_dict[tr], rho_max)
        if dire in ("<=", "==/<="):
            active = 1 * ((w_norm - Q[tr]) > -rhoj)
            Q_star[tr] = active * (w_norm + rho_dict[tr] - Q[tr]) + Q[tr]
        if lbc == 0:
            active = 1 * (wj < rho_dict[tr])
            lb.extend((active * wj).tolist())
        else:
            lb.extend([-np.inf] * len(wj))
        jmin = jmax
    return Q_star, lb


def df_EST(w_constr, w, B, J, KM):
    """Effective degrees of freedom per constraint (scpi ``df_EST``).

    ols: ``J``; lasso: ``#nonzero``; simplex: ``#nonzero - 1``; ridge / L1-L2:
    ``sum_k d_k^2 / (d_k^2 + lambda)`` over the positive singular values of the
    (full, stacked) donor design ``B``. ``KM`` covariate columns are added.
    """
    name = w_constr["name"]
    p = w_constr.get("p")
    dire = w_constr.get("dir")
    if name == "ols" or p == "no norm":
        df = float(J)
    elif name == "lasso" or (p == "L1" and dire == "<="):
        df = float(np.sum(np.abs(w.iloc[:, 0].values) >= 1e-6))
    elif name == "simplex" or (p == "L1" and dire == "=="):
        df = float(np.sum(np.abs(w.iloc[:, 0].values) >= 1e-6) - 1)
    else:  # ridge / L1-L2 / L2
        d = np.linalg.svd(np.asarray(B, float), compute_uv=False)
        d = d[d > 0]
        df = float(np.sum(d ** 2 / (d ** 2 + w_constr["lambda"])))
    return df + KM


def u_des_prep_one(B, C, coig_data, constant, index, index_w):
    """u_order=1, u_lags=0."""
    if coig_data:
        B_diff = B - B.groupby('feature').shift(1)
        u_des_0 = pd.concat([B_diff, C], axis=1).loc[:, index]
    else:
        Z = pd.concat([B, C], axis=1)
        u_des_0 = Z.loc[:, index]
    if constant is False:
        colname = B.columns[0].split('_', 1)[0] + '_0_constant'
        u_des_0.insert(len(u_des_0.columns), colname, np.ones(len(u_des_0)))
    return u_des_0


def trendRemove(x):
    sel = []
    for c in x.columns.tolist():
        cp_ = str(c).split('_')
        sel.append(not (len(cp_) >= 3 and cp_[2] == 'trend'))
    return x.loc[:, np.array(sel)], sel


def e_des_prep_one(B, C, P, res, Y_donors, out_feat, J, index, index_w, coig_data,
                   T0, T1, constant, outcome_var, effect, iota, tr, P_diff_pre=None):
    """e_order=1, e_lags=0, out_feat True."""
    P, selp = trendRemove(P)
    C, selc = trendRemove(C)
    index = np.array(index)[np.array(selp)].tolist()
    Z = pd.concat([B, C], axis=1)
    if P_diff_pre is not None:
        P_diff_pre, _ = trendRemove(P_diff_pre)

    e_res = res.loc[(outcome_var,), ]

    if coig_data:
        B_diff = B - B.groupby('feature').shift(1)
        e_des_0 = pd.concat([B_diff, C], axis=1).loc[:, index]
        if effect == "time":
            P_first = (P.iloc[[0], :J] * iota - B.iloc[T0 - 1, :].values) / iota
        else:
            P_first = P.iloc[[0], :J] - B.iloc[T0 - 1, :].values
        P_diff = P.iloc[:, :J].diff()
        P_diff.iloc[0, :] = P_first
        e_des_1 = pd.concat([P_diff.loc[:, index_w], P.iloc[:, J:]], axis=1)
    else:
        e_des_0 = Z.loc[:, index]
        e_des_1 = P.loc[:, index]

    if P_diff_pre is not None:
        e_des_1 = P_diff_pre.loc[:, index]

    if constant is False:
        e_des_0.insert(len(e_des_0.columns), '0_constant', np.ones(len(e_des_0)))
        if effect == "time":
            e_des_1.insert(len(e_des_1.columns), '0_constant', np.ones(len(e_des_1)) / iota)
        else:
            e_des_1.insert(len(e_des_1.columns), '0_constant', np.ones(len(e_des_1)))

    e_des_0 = e_des_0.loc[(outcome_var,), ]
    return e_res, e_des_0, e_des_1


def DUflexGet(u_des_0, C):
    sel_B = [u not in C.columns.tolist() for u in u_des_0.columns]
    sel_C = [not b for b in sel_B]
    D_b = u_des_0.loc[:, sel_B]
    D_c = u_des_0.loc[:, sel_C]
    f_id = pd.get_dummies(u_des_0.index.get_level_values('feature')).astype(float)
    f_id.set_index(D_b.index, inplace=True)
    D_bb = pd.concat([D_b, f_id], axis=1)
    features = f_id.columns.tolist()
    tomult = D_b.columns.tolist()
    D_b_int = pd.DataFrame(index=D_b.index)
    for f in features:
        aux = D_bb.loc[:, tomult].multiply(D_bb.loc[:, f], axis="index")
        D_b_int = pd.concat([D_b_int, aux], axis=1)
    D_b_int.set_index(D_c.index, inplace=True)
    return pd.concat([D_b_int, D_c], axis=1)


def detectConstant(x, tr, scale_x=1):
    x = x * scale_x
    n = len(x.loc[complete_cases(x), ])
    col_keep = x.sum(axis=0) != n
    col_keep = np.logical_and(col_keep, (x.sum(axis=0) != 0))
    x = x.loc[:, col_keep].copy()
    x.insert(0, tr + "_constant", 1)
    x = x / scale_x
    return x


def avoidCollin(mat_0, mat_1, scale_x, tr):
    if mat_0.shape[1] >= (mat_0.shape[0] - 1):
        nm = f"{tr}_constant"
        mat_0 = pd.DataFrame(1, index=mat_0.index, columns=[nm])
        mat_1 = pd.DataFrame(1.0 / scale_x, index=mat_1.index, columns=[nm])
    return mat_0, mat_1


def df_EST_simplex(w, KM_tot):
    df = sum(abs(w.iloc[:, 0].values) >= 1e-6) - 1
    return df + KM_tot


def u_sigma_est_HC1(u_mean, res, Z, V, TT, df):
    ZZ = np.asarray(Z, dtype=float)
    VV = np.asarray(V, dtype=float)
    vc = TT / (TT - df)
    omega_diag = np.asarray((res - u_mean) ** 2).flatten() * vc
    VZ = VV.dot(ZZ)
    Sigma = (VZ.T * omega_diag).dot(VZ) / (TT ** 2)
    return Sigma


# ---------------------------------------------------------------------------
# Conic in-sample optimisation (mirrors scpi_in_diag, simplex)
# ---------------------------------------------------------------------------
def mat_regularize(Q):
    """Reduce ``Q`` to its non-null factor ``Qreg`` (rows x n) with
    ``Qreg' Qreg = Q / scale`` on the kept subspace, dropping near-null
    eigen-directions. Returns ``(scale, Qreg)``; ``Qreg`` is empty when ``Q`` is
    ~0. Mirrors scpi's ``matRegularize``."""
    Qa = np.asarray(Q, dtype=float)
    n = Qa.shape[0]
    w, V = eigh(Qa)
    cond = 1e6 * np.finfo(float).eps
    scale = float(np.max(np.abs(w))) if w.size else 0.0
    if scale < cond:
        return 0.0, np.zeros((0, n))
    wsc = w / scale
    maskp = wsc > cond
    Qreg = (V[:, maskp] * np.sqrt(wsc[maskp])).T      # (nkeep, n), Qreg'Qreg = Q/scale
    return scale, Qreg


def _ecos_simplex_dims_AB(n, J, Qsum, red):
    """Constant ECOS pieces for the single-unit simplex QCQP: cone dims, the
    sum-to-one equality (A, b). ``ns = 1`` slack (the SOC epigraph ``t``)."""
    ns = 1
    dims = {"l": J + 1, "q": [n + 2 - red]}
    A = np.zeros((1, n + ns))
    A[0, 0:J] = 1.0
    A = sparse.csc_matrix(A)
    b = np.array([Qsum], dtype=float)
    return ns, dims, A, b


def conic_bounds_unit(beta, Q, P_rows, lb, Qsum, zeta_unit, sims):
    """For one treated unit, solve min/max ``p.(beta-x)`` over the localized set
    for every simulation draw and post-treatment horizon, using scpi's exact
    ECOS second-order-cone construction.

    The constraint ``(x-beta)'Q(x-beta) - 2 G'(x-beta) <= 0`` is encoded as a
    rotated SOC with the scaled, dimension-reduced factor ``Qreg`` (see
    :func:`mat_regularize`): the linear part uses the full ``Q`` (via ``beta_q``,
    ``beta_q_beta``) and the cone uses ``Qreg`` / ``scale``. Solving with ``ecos``
    (rather than a cvxpy reformulation) reproduces scpi cell-for-cell, including
    the near-null-direction handling that the per-cell predictands exercise.
    """
    beta = np.asarray(beta, dtype=float)
    n = len(beta)
    J = len(lb)
    Qa = np.asarray(Q, dtype=float)
    lb = np.asarray(lb, dtype=float)

    scale, Qreg = mat_regularize(Qa)
    red = n - Qreg.shape[0]
    ns, dims, A, b = _ecos_simplex_dims_AB(n, J, Qsum, red)

    beta_q = beta.dot(Qa)                 # full Q, as in scpi
    beta_q_beta = float(beta.dot(Qa.dot(beta)))

    # constant G rows (lower bounds + SOC epigraph + quadratic), built once
    G_lb = np.concatenate((-np.eye(J), np.zeros((J, n - J + ns))), axis=1)
    G_soc_t = np.array([[0.0] * n + [-1.0], [0.0] * n + [1.0]])
    G_quad = -2.0 * np.concatenate((Qreg, np.zeros((Qreg.shape[0], ns))), axis=1)
    h_const_tail = np.concatenate(([-ll for ll in lb], [1.0, 1.0],
                                   np.zeros(Qreg.shape[0])))

    res_lb = np.full((sims, len(P_rows)), np.nan)
    res_ub = np.full((sims, len(P_rows)), np.nan)
    OK = ("Optimal solution found", "Close to optimal solution found")

    for s in range(sims):
        Gd = zeta_unit[:, s]
        a = -2.0 * Gd - 2.0 * beta_q
        d = 2.0 * float(Gd.dot(beta)) + beta_q_beta
        G_lin = np.concatenate((a, [scale]))[None, :]
        Gmat = sparse.csc_matrix(np.concatenate((G_lin, G_lb, G_soc_t, G_quad), axis=0))
        h = np.concatenate(([-d], h_const_tail))
        for hor, p in enumerate(P_rows):
            p = np.asarray(p, dtype=float)
            c_lb = np.concatenate((-p, np.zeros(ns)))
            sol = ecos.solve(c_lb, Gmat, h, dims, A=A, b=b, verbose=False)
            if sol["info"]["infostring"] in OK:
                res_lb[s, hor] = float(p.dot(beta - sol["x"][0:n]))
            c_ub = np.concatenate((p, np.zeros(ns)))
            sol = ecos.solve(c_ub, Gmat, h, dims, A=A, b=b, verbose=False)
            if sol["info"]["infostring"] in OK:
                res_ub[s, hor] = float(p.dot(beta - sol["x"][0:n]))
    return res_lb, res_ub


def conic_bounds_unit_family(beta, Q, P_rows, name, Q_star, Q2_star, lb,
                             zeta_unit, sims):
    """cvxpy in-sample conic bounds for one treated unit under the non-simplex
    weight-constraint family (ols / lasso / ridge / L1-L2).

    Solves, for every simulation draw and post-treatment horizon, the min/max of
    ``p.(beta - x)`` over the localised compatible set
    ``{ (x-beta)'Q(x-beta) - 2 G'(x-beta) <= 0 } intersect W``, where ``W`` is the
    constraint's donor weight-set (``x[:J]``): lasso ``||.||1 <= Q_star``; ridge
    ``||.||2 <= Q_star``; L1-L2 ``x>=lb, sum==Q_star, ||.||2<=Q2_star``; ols free.
    This mirrors scpi's ``scpi_in_diag`` construction and the (cross-validated)
    single-unit ``scpi_intervals`` QCQP; the simplex case stays on the exact ECOS
    path (:func:`conic_bounds_unit`).
    """
    beta = np.asarray(beta, dtype=float)
    n = len(beta)
    J = len(lb)
    Qa = np.asarray(Q, dtype=float)
    lb = np.asarray(lb, dtype=float)

    scale, Qreg = mat_regularize(Qa)          # Qreg (k, n); scale * ||Qreg z||^2 = z'Q z
    x = cp.Variable(n)
    c = cp.Parameter(n)
    Gstar = cp.Parameter(n)
    if Qreg.shape[0] == 0:  # pragma: no cover - degenerate Q (no donor variation)
        quad = cp.Constant(0.0)
    else:
        quad = scale * cp.sum_squares(Qreg @ (x - beta))
    cons = [quad - 2.0 * Gstar @ (x - beta) <= 0.0]
    xd = x[:J]
    if name == "lasso":
        cons.append(cp.norm1(xd) <= Q_star)
    elif name == "ridge":
        cons.append(cp.sum_squares(xd) <= Q_star ** 2)
    elif name == "L1-L2":
        cons += [xd >= lb, cp.sum(xd) == Q_star,
                 cp.sum_squares(xd) <= Q2_star ** 2]
    # ols: no weight-set constraint

    prob_min = cp.Problem(cp.Minimize(c @ x), cons)
    prob_max = cp.Problem(cp.Maximize(c @ x), cons)

    def _solve(prob):
        for solver in (cp.CLARABEL, cp.ECOS, cp.SCS):
            try:
                prob.solve(solver=solver, warm_start=True)
                if prob.status in ("optimal", "optimal_inaccurate") \
                        and x.value is not None:
                    return np.asarray(x.value).ravel()
            except Exception:
                continue
        return None

    res_lb = np.full((sims, len(P_rows)), np.nan)
    res_ub = np.full((sims, len(P_rows)), np.nan)
    for s in range(sims):
        Gstar.value = zeta_unit[:, s]
        for hor, p in enumerate(P_rows):
            c.value = np.asarray(p, dtype=float)
            xs = _solve(prob_max)                 # maximise p.x  -> min p.(beta-x)
            if xs is not None:
                res_lb[s, hor] = float(np.asarray(p).dot(beta - xs))
            xs = _solve(prob_min)                 # minimise p.x  -> max p.(beta-x)
            if xs is not None:
                res_ub[s, hor] = float(np.asarray(p).dot(beta - xs))
    return res_lb, res_ub


# ---------------------------------------------------------------------------
# Out-of-sample uncertainty (mirrors scpi_out)
# ---------------------------------------------------------------------------
def cond_pred(y, x, xpreds, method, tau=None):
    if method == 'lm':
        params = sm.OLS(y, x, missing='drop').fit().params
        return np.asarray(xpreds).dot(np.asarray(params))
    elif method == 'qreg':
        qr = sm.QuantReg(endog=np.asarray(y, dtype=float), exog=np.asarray(x, dtype=float))
        pred = np.empty((len(xpreds), 2))
        pred[:, 0] = qr.fit(q=tau[0]).predict(exog=np.asarray(xpreds, dtype=float))
        pred[:, 1] = qr.fit(q=tau[1]).predict(exog=np.asarray(xpreds, dtype=float))
        return pred


def scpi_out(y, x, preds, e_method, alpha, effect):
    idx = preds.index
    y = np.asarray(y, dtype=float)[:, 0]
    x = np.asarray(x, dtype=float)
    preds_arr = np.asarray(preds, dtype=float)

    if e_method in ('gaussian', 'ls'):
        x_more = np.vstack((preds_arr, x))
        fit = cond_pred(y, x, x_more, 'lm')
        e_mean = fit[:len(preds_arr)]
        if effect == "time":
            e_mean = pd.DataFrame(e_mean, index=idx).groupby(level='Time').mean().values[:, 0]
        y_fit = fit[len(preds_arr):]
        y_var = np.log((y - y_fit) ** 2)
        var_pred = cond_pred(y_var, x, x_more, 'lm')
        res_var = var_pred[len(preds_arr):]
        if effect == "time":
            var_pred_p = pd.DataFrame(var_pred[:len(preds_arr)], index=idx).groupby(level='Time').mean().values[:, 0]
        else:
            var_pred_p = var_pred[:len(preds_arr)]
        q_pred = cond_pred(y - y_fit, x, x_more, 'qreg', tau=[0.25, 0.75])
        if effect == "time":
            q3 = pd.DataFrame(q_pred[:len(preds_arr), 1], index=idx).groupby(level='Time').mean().values[:, 0]
            q1 = pd.DataFrame(q_pred[:len(preds_arr), 0], index=idx).groupby(level='Time').mean().values[:, 0]
        else:
            q3 = q_pred[:len(preds_arr), 1]
            q1 = q_pred[:len(preds_arr), 0]
        IQ = np.abs(q3 - q1)

        if e_method == 'gaussian':
            e_sig2 = np.exp(var_pred_p)
            e_sig = np.c_[np.sqrt(e_sig2), IQ / 1.34].min(axis=1)
            eps = np.sqrt(-np.log(alpha) * 2) * e_sig
            lb = (e_mean - eps).reshape(-1, 1)
            ub = (e_mean + eps).reshape(-1, 1)
            e_1, e_2 = e_mean, e_sig2
        else:  # ls
            e_sig = np.sqrt(np.exp(var_pred_p))
            e_sig = np.c_[e_sig, IQ / 1.34].min(axis=1)
            y_st = (y - y_fit) / np.sqrt(np.exp(res_var))
            lb = (e_mean + e_sig * np.quantile(y_st, alpha)).reshape(-1, 1)
            ub = (e_mean + e_sig * np.quantile(y_st, 1 - alpha)).reshape(-1, 1)
            e_1, e_2 = e_mean, e_sig ** 2

    elif e_method == 'qreg':
        e_pred = cond_pred(y, x, preds_arr, 'qreg', tau=[alpha, 1 - alpha])
        if effect == "time":
            lb = pd.DataFrame(e_pred[:, 0], index=idx).groupby(level='Time').mean().values[:, 0].reshape(-1, 1)
            ub = pd.DataFrame(e_pred[:, 1], index=idx).groupby(level='Time').mean().values[:, 0].reshape(-1, 1)
        else:
            lb = e_pred[:, [0]]
            ub = e_pred[:, [1]]
        e_1 = e_2 = None

    if effect == "time":
        idx = idx.unique('Time')
    lb = pd.DataFrame(lb, index=idx)
    ub = pd.DataFrame(ub, index=idx)
    return lb, ub, e_1, e_2


def simultaneousPredGet(vsig, T1, T1_tot, iota, u_alpha, e_alpha, e_lb, e_ub, e_1):
    vsigLB = vsig[:, :T1_tot]
    vsigUB = vsig[:, T1_tot:]
    jmin = 0
    w_lb_joint = []
    w_ub_joint = []
    for i in range(iota):
        jmax = T1[i] + jmin
        lb_joint = np.nanquantile(np.nanmin(vsigLB[:, jmin:jmax], axis=0), q=u_alpha / 2)
        w_lb_joint += [lb_joint] * T1[i]
        ub_joint = np.nanquantile(np.nanmax(vsigUB[:, jmin:jmax], axis=0), q=1 - u_alpha / 2)
        w_ub_joint += [ub_joint] * T1[i]
        jmin = jmax

    eps = 1
    if len(e_1) > 1:
        eps = []
        for i in range(iota):
            eps += [sqrt(log(T1[i] + 1))] * T1[i]

    e_lb_joint = np.array(e_lb[0].values) * np.array(eps)
    e_ub_joint = np.array(e_ub[0].values) * np.array(eps)
    ML = e_lb_joint + np.array(w_lb_joint)
    MU = e_ub_joint + np.array(w_ub_joint)
    return ML, MU


# ===========================================================================
# Main orchestration  (mirrors scpi)
# ===========================================================================
def scpi(est, sims=200, u_alpha=0.05, e_alpha=0.05, rho='type-2', rho_max=0.2,
         seed=8894, tsua_double_divide=False):
    treated_units = est['treated_units']
    effect = est['effect']
    outcome_var = est['outcome_var']
    iota = est['iota']
    blocks = est['blocks']
    A, B, C, P, Z = est['A'], est['B'], est['C'], est['P'], est['Z']
    b = est['b']
    w = est['w']
    res = est['res']
    Y_post_fit = est['Y_post_fit']

    # V = identity ("separate")
    V = pd.DataFrame(np.identity(len(B)), index=B.index,
                     columns=B.index.get_level_values('ID'))

    J = {tr: blocks[tr]['J'] for tr in treated_units}
    KM = {tr: blocks[tr]['KM'] for tr in treated_units}
    KMI = sum(KM.values())
    T0_M = {tr: sum(blocks[tr]['T0_features'].values()) for tr in treated_units}
    T0 = {tr: blocks[tr]['T0_features'] for tr in treated_units}
    T1 = {tr: blocks[tr]['T1'] for tr in treated_units}
    coig = {tr: blocks[tr]['cointegrated_data'] for tr in treated_units}
    constant = {tr: blocks[tr]['glob_cons'] for tr in treated_units}
    out_feat = {tr: blocks[tr]['out_in_features'] for tr in treated_units}
    if effect == "unit":
        T1 = {tr: 1 for tr in treated_units}

    # per-unit dictionaries
    A_d = {tr: A.loc[pd.IndexSlice[tr, :, :]] for tr in treated_units}
    res_d = {tr: res.loc[pd.IndexSlice[tr, :, :]] for tr in treated_units}
    B_d = mat2dict(B, treated_units, cols=True)
    C_d = mat2dict(C, treated_units, cols=True)
    Z_d = mat2dict(Z, treated_units, cols=True)
    Yd_d = mat2dict(est['Y_donors'], treated_units, cols=True)
    w_d = {tr: w.loc[pd.IndexSlice[tr, :]] for tr in treated_units}
    if effect == "time":
        P_d = {}
        for tr in treated_units:
            csel = [c.split('_')[0] == tr for c in P.columns.tolist()]
            P_d[tr] = P.loc[:, np.array(csel)]
    else:
        P_d = mat2dict(P, treated_units, cols=True)
    Pd_d = {}
    if est['P_diff'] is not None:
        Pd_all = est['P_diff']
        for tr in treated_units:
            if tr in Pd_all.index.get_level_values(0):
                csel = [c.split('_')[0] == tr for c in Pd_all.columns.tolist()]
                sub = Pd_all.loc[pd.IndexSlice[tr, :], np.array(csel)]
                if 'ID' in (sub.index.names or []):
                    sub = sub.reset_index(level=0, drop=True)
                Pd_d[tr] = sub

    col_order = Z.columns.tolist()

    # weight-constraint family: scest records the per-unit normalised spec. The
    # constraint name is uniform across treated units (scpi applies one spec);
    # ``constr_name == "simplex"`` keeps the validated ECOS conic path, everything
    # else routes through the cvxpy family conic.
    from copy import deepcopy
    w_constr_inf = est.get('w_constr_inf') or {
        tr: {"name": "simplex", "p": "L1", "dir": "==", "lb": 0.0, "Q": 1.0,
             "Q2": None, "lambda": None} for tr in treated_units}
    constr_name = w_constr_inf[treated_units[0]]["name"]
    wc_orig = {tr: deepcopy(w_constr_inf[tr]) for tr in treated_units}

    rho_dict = {}
    iw_dict = {}
    Qstar_pre = {}
    Q2star_pre = {}
    store = {}
    for tr in treated_units:
        wc_aux = deepcopy(w_constr_inf[tr])
        _wc, index_w, rho_tr, Qstar_pre[tr], Q2star_pre[tr] = local_geom(
            wc_aux, rho, rho_max, res_d[tr], B_d[tr],
            T0_M[tr], J[tr], KM[tr], w_d[tr])
        rho_dict[tr] = rho_tr
        iw_dict[tr] = index_w
        index_i = index_w.tolist() + [True] * KM[tr]

        ud0 = u_des_prep_one(B_d[tr], C_d[tr], coig[tr], constant[tr], index_i, index_w)
        er, ed0, ed1 = e_des_prep_one(B_d[tr], C_d[tr], P_d[tr], res_d[tr],
                                      Yd_d[tr], out_feat[tr], J[tr], index_i, index_w,
                                      coig[tr], T0[tr][outcome_var], T1[tr],
                                      constant[tr], outcome_var, effect, iota, tr,
                                      Pd_d.get(tr))

        # NaN removal - in sample
        X = pd.concat([A_d[tr], res_d[tr], ud0, Z_d[tr]], axis=1)
        tosel = complete_cases(X)
        X_na = X.loc[tosel, ]
        j2, j3, j4 = 1, 2, 2 + len(ud0.columns)
        r_na = X_na.iloc[:, j2:j3]
        ud0_na = X_na.iloc[:, j3:j4]
        Z_na = X_na.iloc[:, j4:]

        # NaN removal - out of sample
        X2 = pd.concat([er, ed0], axis=1)
        tosel2 = complete_cases(X2)
        X2_na = X2.loc[tosel2, ]
        er_na = X2_na.iloc[:, [0]]
        ed0_na = X2_na.iloc[:, 1:]

        P_na = P_d[tr].loc[complete_cases(P_d[tr]), ]

        for df_ in (ud0_na, r_na, er_na, ed0_na, Z_na):
            df_.insert(0, 'ID', tr)
            df_.set_index('ID', append=True, drop=True, inplace=True)
        if effect == "time":
            ix = pd.MultiIndex.from_product([[tr], ed1.index.get_level_values(0).tolist()],
                                            names=['ID', 'Time'])
            ed1 = ed1.copy()
            ed1.index = ix
        else:
            ed1 = ed1.copy()
            ed1.insert(0, 'ID', tr)
            ed1.set_index('ID', append=True, drop=True, inplace=True)
            P_na = P_na.copy()
            P_na.insert(0, 'ID', tr)
            P_na.set_index('ID', append=True, drop=True, inplace=True)

        store[tr] = dict(ud0_na=ud0_na, r_na=r_na, er_na=er_na, ed0_na=ed0_na,
                         ed1=ed1, Z_na=Z_na, P_na=P_na, index_w=index_w)

    # stack
    Z_na = pd.concat([store[tr]['Z_na'] for tr in treated_units], axis=1).fillna(0)
    res_na = pd.concat([store[tr]['r_na'] for tr in treated_units], axis=0).fillna(0)
    u_des_0_na = pd.concat([store[tr]['ud0_na'] for tr in treated_units], axis=1).fillna(0)
    e_res_na = pd.concat([store[tr]['er_na'] for tr in treated_units], axis=0).fillna(0)
    e_des_0_na = pd.concat([store[tr]['ed0_na'] for tr in treated_units], axis=1).fillna(0)
    e_des_1 = pd.concat([store[tr]['ed1'] for tr in treated_units], axis=1).fillna(0)

    u_des_0_na = u_des_0_na.reorder_levels(['ID', 'feature', 'Time'])
    e_des_0_na = e_des_0_na.reorder_levels(['ID', 'Time'])
    e_des_1 = e_des_1.reorder_levels(['ID', 'Time'])
    e_res_na = e_res_na.reorder_levels(['ID', 'Time'])
    res_na = res_na.reorder_levels(['ID', 'feature', 'Time'])
    Z_na = Z_na.reorder_levels(['ID', 'feature', 'Time'])
    Z_na = Z_na[col_order]

    TT = len(Z_na)
    V_na = np.identity(TT)

    # two-step localisation -> per-unit norm budget Q_star (=1 for simplex) and
    # per-donor lower bounds lb (pinned for simplex/L1-L2, -inf for signed
    # constraints). scpi feeds localgeom2step the pre-mutation constraint Q.
    Q_for_2step = {}
    for tr in treated_units:
        Q_for_2step[tr] = (wc_orig[tr]["Q2"] if wc_orig[tr].get("p") == "L1-L2"
                           else wc_orig[tr]["Q"])
    Q_star, lb = localgeom2step(w, rho_dict, w_constr_inf, Q_for_2step,
                                treated_units, J, rho_max=rho_max)
    if constr_name == "L1-L2":
        Q2_star = deepcopy(Q_star)
        Q_star = {tr: wc_orig[tr]["Q"] for tr in treated_units}
    else:
        Q2_star = {tr: Q2star_pre[tr] for tr in treated_units}

    # ----- u_mean -----
    u_des_dict = mat2dict(u_des_0_na, treated_units, cols=False)
    u_flex_blocks = []
    u_noflex_blocks = []
    for tr in treated_units:
        ufl = DUflexGet(u_des_dict[tr], C_d[tr])
        ufl.insert(0, 'ID', tr)
        ufl.set_index('ID', append=True, drop=True, inplace=True)
        u_flex_blocks.append(ufl)
        unf = u_des_dict[tr].copy()
        unf.insert(0, 'ID', tr)
        unf.set_index('ID', append=True, drop=True, inplace=True)
        u_noflex_blocks.append(unf)
    u_flex = pd.concat(u_flex_blocks, axis=1)
    u_noflex = pd.concat(u_noflex_blocks, axis=1)

    T_u = len(u_des_0_na)
    df_U = T_u - 10
    if df_U <= len(u_noflex.columns):
        u_des_use = pd.DataFrame([1] * T_u, index=u_des_0_na.index)
    elif df_U <= len(u_flex.columns):
        u_des_use = u_noflex.reorder_levels(['ID', 'feature', 'Time']).fillna(0)
    else:
        u_des_use = u_flex.reorder_levels(['ID', 'feature', 'Time']).fillna(0)

    u_mean = LinearRegression().fit(np.asarray(u_des_use, dtype=float),
                                    np.asarray(res_na, dtype=float)).predict(
                                        np.asarray(u_des_use, dtype=float))

    # ----- Sigma -----
    # scpi computes df once, on the full stacked weights/donor design, with the
    # (uniform) constraint spec; for ridge/L1-L2 it uses the last unit's lambda.
    Jtot = sum(J.values())
    df = df_EST(w_constr_inf[treated_units[-1]], w, np.asarray(B, dtype=float),
                Jtot, KMI)
    if df >= TT:
        df = TT - 1
    Sigma = u_sigma_est_HC1(u_mean, res_na, Z_na, V_na, TT, df)
    Sigma_root = sqrtm(Sigma).real

    # ----- in-sample conic -----
    beta_full = np.asarray(b, dtype=float).flatten()
    np.random.seed(seed)
    zeta_raw = np.random.normal(0, 1, size=(len(beta_full) * sims)).reshape(len(beta_full), sims)
    zeta = Sigma_root @ zeta_raw

    Z_na_arr = np.asarray(Z_na, dtype=float)
    Q_full = Z_na_arr.T.dot(Z_na_arr) / TT
    zcols = Z_na.columns.tolist()
    # beta (=b) and zeta share Z_na's column ordering: [all donors, all covs].
    # Gather each unit's coordinates by column prefix (mirrors scpi mat2dict),
    # NOT by contiguous slicing.
    col_pref = [c.split('_')[0] for c in zcols]

    # per-unit lower bounds (donor block only), in donor order of w
    lb_d = {}
    jmin = 0
    for tr in treated_units:
        lb_d[tr] = lb[jmin:jmin + J[tr]]
        jmin += J[tr]

    vsig_lb_list = []
    vsig_ub_list = []
    for tr in treated_units:
        sel = np.array([p == tr for p in col_pref])
        Qi = Q_full[np.ix_(sel, sel)]
        beta_i = beta_full[sel]
        zeta_i = zeta[sel, :]
        P_na = store[tr]['P_na']
        if effect == "time":
            csel = [c.split('_')[0] == tr for c in P_na.columns.tolist()]
            P_na_tr = P_na.loc[:, np.array(csel)]
        else:
            P_na_tr = P_na
        P_rows = [np.asarray(P_na_tr.iloc[h, :], dtype=float) for h in range(len(P_na_tr))]
        if constr_name == "simplex":
            rlb, rub = conic_bounds_unit(beta_i, Qi, P_rows, lb_d[tr],
                                         Q_star[tr], zeta_i, sims)
        else:
            rlb, rub = conic_bounds_unit_family(
                beta_i, Qi, P_rows, constr_name, Q_star[tr],
                Q2_star.get(tr), lb_d[tr], zeta_i, sims)
        vsig_lb_list.append(rlb)
        vsig_ub_list.append(rub)

    if effect == "time":
        # scpi divides the per-unit simulated in-sample bound by iota here, on
        # top of the P/iota scaling already applied in scdataMulti -> a 1/iota^2
        # shrinkage. Set tsua_double_divide=False for the statistically correct
        # 1/iota scaling (average of iota per-unit in-sample errors).
        div = iota if tsua_double_divide else 1.0
        vsig_lb = np.nan_to_num(vsig_lb_list[0], nan=0) / div
        vsig_ub = np.nan_to_num(vsig_ub_list[0], nan=0) / div
        for k in range(1, len(treated_units)):
            vsig_lb = vsig_lb + np.nan_to_num(vsig_lb_list[k], nan=0) / div
            vsig_ub = vsig_ub + np.nan_to_num(vsig_ub_list[k], nan=0) / div
    else:
        vsig_lb = np.concatenate(vsig_lb_list, axis=1)
        vsig_ub = np.concatenate(vsig_ub_list, axis=1)
    vsig = np.concatenate([vsig_lb, vsig_ub], axis=1)

    nP = vsig_lb.shape[1]
    w_lb = np.nanquantile(vsig[:, :nP], u_alpha / 2, axis=0)
    w_ub = np.nanquantile(vsig[:, nP:], 1 - u_alpha / 2, axis=0)

    yfit = np.asarray(Y_post_fit, dtype=float).flatten()
    sc_l_0 = yfit + w_lb
    sc_r_0 = yfit + w_ub

    # ----- out-of-sample design rebuild -----
    ed0_dict = mat2dict(e_des_0_na, treated_units, cols=True)
    ed1_dict = mat2dict(e_des_1, treated_units, cols=True)
    T_e = len(e_des_0_na)
    params_e = len(e_des_0_na.columns)
    if (T_e - 10) <= params_e:
        for tr in treated_units:
            ed0_dict[tr] = pd.DataFrame(np.ones(len(ed0_dict[tr])), columns=[tr],
                                        index=ed0_dict[tr].index)
            scale = iota if effect == "time" else 1
            ed1_dict[tr] = pd.DataFrame(np.ones(len(ed1_dict[tr])) / scale,
                                        columns=[f"{tr}_constant"], index=ed1_dict[tr].index)

    scale_x = iota if effect == "time" else 1
    e0_blocks, e1_blocks = [], []
    for tr in treated_units:
        e0 = detectConstant(ed0_dict[tr], tr)
        e1 = detectConstant(ed1_dict[tr], tr, scale_x)
        e0, e1 = avoidCollin(e0, e1, scale_x, tr)
        e0.insert(0, 'ID', tr); e0.set_index('ID', drop=True, inplace=True)
        e1.insert(0, 'ID', tr); e1.set_index('ID', drop=True, inplace=True)
        e0_blocks.append(e0); e1_blocks.append(e1)
    e_des_0_use = pd.concat(e0_blocks, axis=0)
    e_des_1_use = pd.concat(e1_blocks, axis=0)
    e_des_0_use.index = e_res_na.index
    if effect == "time":
        nP_time = len(store[treated_units[0]]['P_na'])
        idx = pd.MultiIndex.from_product([treated_units, list(range(1, nP_time + 1))],
                                         names=['ID', 'Time'])
        e_des_1_use.index = idx
    else:
        Pidx = pd.concat([store[tr]['P_na'] for tr in treated_units], axis=0).index
        e_des_1_use.index = Pidx
    e_des_0_use = e_des_0_use.fillna(0)
    e_des_1_use = e_des_1_use.fillna(0)

    # out-of-sample bounds
    e_lb_g, e_ub_g, e1m, e2v = scpi_out(e_res_na, e_des_0_use, e_des_1_use,
                                        'gaussian', e_alpha / 2, effect)
    e_lb_ls, e_ub_ls, _, _ = scpi_out(e_res_na, e_des_0_use, e_des_1_use,
                                      'ls', e_alpha / 2, effect)
    e_lb_q, e_ub_q, _, _ = scpi_out(e_res_na, e_des_0_use, e_des_1_use,
                                    'qreg', e_alpha / 2, effect)

    sc_l_1 = sc_l_0 + e_lb_g.iloc[:, 0].values
    sc_r_1 = sc_r_0 + e_ub_g.iloc[:, 0].values

    # joint
    if effect == "unit-time":
        T1val = [T1[tr] for tr in treated_units]
        ML, MU = simultaneousPredGet(vsig, T1val, nP, iota, u_alpha, e_alpha,
                                     e_lb_g, e_ub_g, e1m)
    else:
        ML, MU = simultaneousPredGet(vsig, [nP], nP, 1, u_alpha, e_alpha,
                                     e_lb_g, e_ub_g, e1m)

    return dict(Y_post_fit=yfit, w_lb=w_lb, w_ub=w_ub,
                sc_l_0=sc_l_0, sc_r_0=sc_r_0, sc_l_1=sc_l_1, sc_r_1=sc_r_1,
                e_lb_g=e_lb_g.iloc[:, 0].values, e_ub_g=e_ub_g.iloc[:, 0].values,
                e_lb_ls=e_lb_ls.iloc[:, 0].values, e_ub_ls=e_ub_ls.iloc[:, 0].values,
                e_lb_q=e_lb_q.iloc[:, 0].values, e_ub_q=e_ub_q.iloc[:, 0].values,
                ML=ML, MU=MU, Sigma=Sigma, index=Y_post_fit.index)


# ===========================================================================
# Public entry point: outcome-only staggered VanillaSC prediction intervals
# ===========================================================================
def _normalize_spec(features, cov_adj, feature_constant, cointegrated,
                    outcome, treated_units):
    """Map a user spec (outcome-only, shared, or per-unit) to the engine's
    ``scdata_multi`` argument forms. ``features`` / ``cov_adj`` may be a list
    (shared across units) or a ``{unit: ...}`` dict; ``feature_constant`` /
    ``cointegrated`` may be a bool or a per-unit dict."""
    if features is None:
        return ({tr: [outcome] for tr in treated_units},
                {tr: None for tr in treated_units}, False, False)
    feats = ({"features": list(features)}
             if isinstance(features, (list, tuple)) else dict(features))
    if cov_adj is None:
        cadj = {tr: None for tr in treated_units}
    elif isinstance(cov_adj, (list, tuple)):
        cadj = {"cov_adj": list(cov_adj)}
    else:
        cadj = dict(cov_adj)
    return feats, cadj, feature_constant, cointegrated


def staggered_pi_bands(
    df,
    *,
    outcome,
    unitid,
    time,
    treat,
    effect="time",
    sims=200,
    u_alpha=0.05,
    e_alpha=0.05,
    seed=8894,
    scpi_compat=False,
    features=None,
    cov_adj=None,
    feature_constant=False,
    cointegrated=False,
    w_constr="simplex",
):
    """Staggered synthetic-control prediction intervals for one predictand.

    Drives the clean-room engine on a long panel and returns the synthetic
    (counterfactual) point estimates and prediction-interval bands. Outcome-only
    by default (simplex weights, never-treated donor pool); pass ``features`` /
    ``cov_adj`` / ``feature_constant`` / ``cointegrated`` to reproduce scpi's
    covariate (multi-feature) staggered specification.

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel with unit, time, outcome and treatment-indicator columns.
    outcome, unitid, time, treat : str
        Column names. ``treat`` is the 0/1 staggered treatment indicator.
    effect : {"time", "unit", "unit-time"}
        The causal predictand: TSUA (event-time, unit-averaged), TAUS
        (per-unit time-average) or TSUS (unit x period).
    sims : int
        Gaussian draws for the in-sample conic simulation.
    u_alpha, e_alpha : float
        In-sample / out-of-sample miscoverage levels.
    seed : int
        Seed for the in-sample Gaussian draws.
    scpi_compat : bool
        When True, reproduce ``scpi``'s ``1 / iota**2`` in-sample scaling of the
        time-aggregated band; when False (default), the statistically correct
        ``1 / iota``. Only affects ``effect="time"``.
    features : list or {unit: list}, optional
        Matching features (outcome-like series). A list is shared across units;
        a dict gives per-unit feature sets. ``None`` -> outcome-only.
    cov_adj : list or {unit: ...}, optional
        Covariate-adjustment terms (e.g. ``["constant", "trend"]``), shared or
        per-unit.
    feature_constant : bool or {unit: bool}
        Whether to include a global constant (scpi's ``constant``).
    cointegrated : bool or {unit: bool}
        Whether to difference the data for cointegration (scpi's
        ``cointegrated_data``).

    Returns
    -------
    dict
        Keys: ``index`` (predictand row labels), ``point`` (synthetic fit),
        ``insample_lb`` / ``insample_ub`` (in-sample-only band), ``lb`` / ``ub``
        (full sub-Gaussian band), ``joint_lb`` / ``joint_ub`` (uniform band),
        and ``Sigma``.
    """
    ren = {unitid: "country", time: "year", treat: "status"}
    data = df.rename(columns=ren).copy()
    treated_units = sorted(
        data.loc[data["status"] == 1, "country"].unique().tolist())
    feats, cadj, const, coint = _normalize_spec(
        features, cov_adj, feature_constant, cointegrated, outcome, treated_units)

    md = scdata_multi(data, outcome, feats, cadj,
                      constant=const, cointegrated_data=coint, effect=effect)
    est = scest(md, w_constr)
    out = scpi(est, sims=sims, u_alpha=u_alpha, e_alpha=e_alpha, seed=seed,
               tsua_double_divide=scpi_compat)

    return {
        "index": out["index"],
        "point": np.asarray(out["Y_post_fit"], dtype=float),
        "insample_lb": np.asarray(out["sc_l_0"], dtype=float),
        "insample_ub": np.asarray(out["sc_r_0"], dtype=float),
        "lb": np.asarray(out["sc_l_1"], dtype=float),
        "ub": np.asarray(out["sc_r_1"], dtype=float),
        "joint_lb": np.asarray(out["Y_post_fit"], dtype=float) + np.asarray(out["ML"], dtype=float),
        "joint_ub": np.asarray(out["Y_post_fit"], dtype=float) + np.asarray(out["MU"], dtype=float),
        "Sigma": out["Sigma"],
    }

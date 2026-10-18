"""The numbers Lv & Li (2026) print, for the study to match.

arXiv:2610.15553v1. Table 1 (p. 26) is Example 1; Table 4 (p. 29) and the text
of Section 6 (pp. 29-30) are the Basque application.
"""

#: Table 1, T1 = 60, T2 = 30, N = 25: (bias, MAE, MSE x 100) by case, method
#: and alpha_0. The two alpha_0 columns are independent Monte Carlo runs of one
#: quantity (see ``simulate``), so their spread is the paper's own Monte Carlo
#: noise.
TABLE1_T60 = {
    "a": {"RRMA": {0: (-0.010, 0.227, 7.568), 1: (-0.003, 0.213, 7.165)},
          "EW":   {0: (-0.011, 0.258, 10.15), 1: (-0.005, 0.253, 10.661)},
          "SBIC": {0: (-0.012, 0.247, 9.224), 1: (-0.005, 0.239, 9.478)},
          "PHCW": {0: (-0.021, 0.237, 9.244), 1: (0.009, 0.234, 9.042)},
          "MSCM": {0: (-0.015, 0.230, 8.301), 1: (0.006, 0.215, 7.231)},
          "LSMA": {0: (-0.020, 0.246, 9.254), 1: (0.014, 0.234, 8.666)}},
    "b": {"RRMA": {0: (0.002, 0.258, 10.43), 1: (-0.001, 0.247, 9.528)},
          "EW":   {0: (0.001, 0.301, 14.281), 1: (0.004, 0.290, 12.89)},
          "SBIC": {0: (0.001, 0.292, 13.44), 1: (0.003, 0.281, 12.15)},
          "PHCW": {0: (-0.001, 0.339, 17.78), 1: (0.009, 0.316, 15.90)},
          "MSCM": {0: (0.014, 0.289, 13.08), 1: (0.014, 0.303, 15.01)},
          "LSMA": {0: (0.001, 0.324, 16.67), 1: (0.018, 0.334, 18.17)}},
    "c": {"RRMA": {0: (0.008, 0.227, 8.100), 1: (-0.003, 0.230, 8.316)},
          "EW":   {0: (0.010, 0.270, 11.64), 1: (0.003, 0.255, 10.40)},
          "SBIC": {0: (0.010, 0.258, 10.59), 1: (0.001, 0.244, 9.529)},
          "PHCW": {0: (0.015, 0.295, 14.25), 1: (0.013, 0.307, 15.44)},
          "MSCM": {0: (-0.008, 0.258, 10.66), 1: (0.005, 0.277, 12.27)},
          "LSMA": {0: (0.008, 0.309, 15.70), 1: (0.018, 0.311, 15.23)}},
}

#: Table 1, T1 = 120, T2 = 60, N = 45, the same layout. Not run by the study
#: (each replication needs about four minutes); kept for a later run.
TABLE1_T120 = {
    "a": {"RRMA": {0: (0.006, 0.151, 3.564), 1: (-0.010, 0.155, 3.774)},
          "EW":   {0: (0.016, 0.249, 9.284), 1: (0.005, 0.238, 8.778)},
          "MSCM": {0: (0.008, 0.150, 3.574), 1: (-0.013, 0.152, 3.751)}},
    "b": {"RRMA": {0: (0.004, 0.177, 5.201), 1: (-0.002, 0.177, 4.908)},
          "EW":   {0: (0.001, 0.254, 10.67), 1: (-0.007, 0.264, 10.557)},
          "MSCM": {0: (-0.011, 0.212, 7.003), 1: (-0.005, 0.195, 6.268)}},
    "c": {"RRMA": {0: (-0.002, 0.145, 3.479), 1: (0.007, 0.150, 3.574)},
          "EW":   {0: (0.011, 0.227, 8.283), 1: (0.020, 0.238, 8.869)},
          "MSCM": {0: (0.001, 0.175, 4.935), 1: (0.007, 0.184, 5.539)}},
}

#: Table 4: (lower, upper) by method and level.
TABLE4 = {
    "RRMA": {0.90: (-0.8027, -0.3925), 0.95: (-0.8420, -0.3532), 0.99: (-0.9188, -0.2764)},
    "EW":   {0.90: (-1.4614, -0.4893), 0.95: (-1.5545, -0.3962), 0.99: (-1.7365, -0.2142)},
    "SBIC": {0.90: (-1.3193, -0.5102), 0.95: (-1.3968, -0.4327), 0.99: (-1.5483, -0.2812)},
    "MSCM": {0.90: (-1.0638, -0.5413), 0.95: (-1.1083, -0.5089), 0.99: (-1.1806, -0.4325)},
}

#: Section 6, p. 29: Algorithm 1 with M_min = 2, M_max = 7 gives S = 8 pairs,
#: in the paper's X_j numbering (the 16 donors in region-number order).
BASQUE_M_RANGE = range(2, 8)
BASQUE_GROUPS = [[1, 5], [2, 3], [4, 6], [7, 8], [9, 10], [11, 12], [14, 15], [13, 16]]
#: p. 30: "RRMA assigns the positive weights to M1, M4, M5 and M8".
BASQUE_SUPPORT = [1, 4, 5, 8]


def implied_rrma():
    """The point estimate and standard error equation (15) implies for Table 4.

    The interval is symmetric, so its midpoint is the estimate at every level
    and its half-width over ``z`` is the standard error.
    """
    from scipy.stats import norm

    mids, ses = [], []
    for level, (lo, hi) in TABLE4["RRMA"].items():
        mids.append((lo + hi) / 2)
        ses.append((hi - lo) / 2 / norm.ppf(0.5 + level / 2))
    return dict(att=sum(mids) / len(mids), att_spread=max(mids) - min(mids),
                se=sum(ses) / len(ses), se_spread=max(ses) - min(ses))

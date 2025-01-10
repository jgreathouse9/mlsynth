from scipy.optimize import lsq_linear
import numpy as np


def step2(R1t, R2t, Rt, b_MSC_c, q1t, q2t, qt, t1, x1, y1, nb, n, bm_MSC_c):

    d1t = np.dot(R1t, b_MSC_c) - q1t
    d2t = np.dot(R2t, b_MSC_c) - q2t
    dt = np.dot(Rt, b_MSC_c) - qt
    test1 = t1 * np.dot(d1t.T, d1t)
    test2 = t1 * np.dot(d2t.T, d2t)

    z1 = np.hstack((x1, y1.reshape(-1, 1)))
    bm_MSC_c_sum0 = np.zeros(nb)
    V_hatI = np.zeros((2, 2))
    d1t_s = np.zeros(nb)
    d2t_s = np.zeros(nb)
    test1_s = np.zeros(nb)
    test2_s = np.zeros(nb)

    for g in range(nb):
        m = t1
        zm = z1[np.random.choice(z1.shape[0], m, replace=True), :]
        ym = zm[:, -1]
        xm = zm[:, :-1]

        lb = np.zeros(n)
        lb[0] = -np.inf
        bm_MSC_c[:, g] = lsq_linear(
            xm, ym, bounds=(lb, np.inf), method="trf", lsmr_tol="auto"
        ).x

        bm_MSC_c_g = bm_MSC_c[:, g]
        bm_MSC_c_sum0[g] = np.sum(bm_MSC_c_g[1:])

        dt_s = np.dot(Rt, bm_MSC_c_g) - qt
        dt_ss = np.dot(Rt, (bm_MSC_c_g - b_MSC_c))
        V_hatI += (m / nb) * np.outer(dt_ss, dt_ss)

        d1t_s[g] = np.dot(R1t, (bm_MSC_c_g - b_MSC_c))[0]
        d2t_s[g] = np.dot(R2t, (bm_MSC_c_g - b_MSC_c))[0]

        test1_s[g] = m * np.dot(d1t_s[g].T, d1t_s[g])
        test2_s[g] = m * np.dot(d2t_s[g].T, d2t_s[g])

    V_hat = np.linalg.inv(V_hatI)

    Js_test = np.zeros(nb)
    for ggg in range(nb):
        ds = np.dot(Rt, (bm_MSC_c[:, ggg] - b_MSC_c))
        Js_test[ggg] = m * np.dot(ds.T, np.dot(V_hat, ds))

    dt = np.dot(Rt, b_MSC_c.reshape(-1, 1)) - qt
    J_test = t1 * np.dot(dt.T, np.dot(V_hat, dt))

    # Calculate p-values
    pJ = np.mean(
        J_test < Js_test
    )  # p-value for joint hypothesis H0. If fail to reject, use Original SC in Step 2. If reject, then look at p1
    p1 = np.mean(
        test1 < test1_s
    )  # p-value for single restriction hypothesis test of sum to one H0a. If fail to reject, use MSCa in Step 2. If reject, then look at p2
    p2 = np.mean(
        test2 < test2_s
    )  # p-value for single restriction hypothesis test of zero intercept H0b. If fail to reject, use MSCb in step 2. Otherwise, use MSC in step 2.
    # Check p-values and recommend SCM model
    if pJ >= 0.05:
        recommended_model = "MSCc"
    elif pJ < 0.05 and p1 >= 0.05 and p2 >= 0.05:
        recommended_model = "MSCa"
    elif pJ < 0.05 and (p1 < 0.05 or p2 < 0.05):
        recommended_model = "MSCb"
    else:
        recommended_model = "SC"

    return recommended_model

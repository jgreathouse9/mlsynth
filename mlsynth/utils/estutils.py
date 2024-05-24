import numpy as np
import cvxpy as cp
from mlsynth.utils.denoiseutils import universal_rank
from mlsynth.utils.resultutils import effects


def ci_bootstrap(b, Nco, x, y, t1, nb, att, method, y_counterfactual):
    """
    Perform subsampling bootstrap.

    Args:
        x (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        t1 (int): Number of observations for the treatment group.
        nb (int): Number of bootstrap samples.

    Returns:
        numpy.ndarray: Array of treatment effects estimates.
    """

    t = len(y_counterfactual)
    m = t1-5

    t2 =  t-t1

    zz1 = np.concatenate((x[:t1], y[:t1].reshape(-1, 1)), axis=1)
    # Confidence Intervals
    sigma_MSC = np.sqrt(np.mean((y[:t1] - y_counterfactual[:t1]) ** 2))

    sigma2_v = np.mean((y[t1:] - y_counterfactual[t1:] - np.mean(y[t1:] - y_counterfactual[t1:])) ** 2)  # \hat \Sigma_v

    e1_star = np.sqrt(sigma2_v) * np.random.randn(t2, nb)  # e_{1t}^* iid N(0, \Sigma^2_v)
    A_star = np.zeros(nb)

    x2 = x[t1 + 1: t, :]

    np.random.seed(1476)

    for g in range(nb):

        np.random.shuffle(zz1)  # the random generator does not repeat the same number

        zm = zz1[:m, :]  # randomly select m rows from z1_{T_1 by (N+1)}
        xm = zm[:, :-1]  # it uses the last m observation
        ym = np.dot(xm, b) + sigma_MSC * np.random.randn(m)

        if method in ["MSCa"]:
            xm = xm[:, 1:]

        if method in ["MSCc"]:
            xm = xm[:, 1:]

        bm = Opt.SCopt(Nco, ym[:t1], t1, xm[:t1], model=method)

        A1_star = -np.mean(np.dot(x2, (bm - b))) * np.sqrt((t2 * m) / t1)  # A_1^*
        A2_star = np.sqrt(t2) * np.mean(e1_star[:, g])
        A_star[g] = A1_star + A2_star  # A^* = A_1^* + A_2^*

    # end of the subsampling-bootstrap loop
    ATT_order = np.sort(A_star / np.sqrt(t2))  # sort subsampling ATT by ascending order

    c005 = ATT_order[int(0.005 * nb)]  # compute critical values
    c025 = ATT_order[int(0.025 * nb)]
    c05 = ATT_order[int(0.05 * nb)]
    c10 = ATT_order[int(0.10 * nb)]
    c25 = ATT_order[int(0.25 * nb)]
    c75 = ATT_order[int(0.75 * nb)]
    c90 = ATT_order[int(0.90 * nb)]
    c95 = ATT_order[int(0.95 * nb)]
    c975 = ATT_order[int(0.975 * nb)]
    c995 = ATT_order[int(0.995 * nb)]
    cr_min = att - ATT_order[nb - 1]  # ATT_order[nb] is the maximum A^* value
    cr_max = att - ATT_order[0]  # ATT_order[1] is the maximum A^* value

    # 95% confidence interval of ATT is [cr_025, cr_975]
    # 90% confidence interval of ATT is [cr_05, cr_95], etc.
    cr_005_995 = [att - c995, att - c005]

    cr_025_0975 = [att - c975, att - c025]

    return cr_025_0975

def TSEST(x, y, t1, nb, donornames, t2):
    # Define a list to store dictionaries for each method
    fit_dicts_list = []
    fit_results = {}
    # List of methods to loop over
    methods = ["SIMPLEX", "MSCb", "MSCa", "MSCc"]

    for method in methods:

        if method in ["MSCc"]:
            x = x[:, 1:]

        Nco = x.shape[1]

        b = Opt.SCopt(Nco, y[:t1], t1, x[:t1], model=method)

        weights_dict = {donor: weight for donor, weight in zip(donornames, np.round(b, 4)) if weight > 0.001}

        if method in ["MSCa"]:
            x = np.c_[np.ones((x.shape[0], 1)), x]

        if method in ["MSCc"]:
            x = np.c_[np.ones((x.shape[0], 1)), x]

        # Calculate the counterfactual outcome
        y_counterfactual = x.dot(b)

        attdict, fitdict, Vectors = effects.calculate(y, y_counterfactual, t1, t2)

        att = attdict["ATT"]

        cis = ci_bootstrap(b, Nco, x, y, t1, nb, att, method, y_counterfactual)

        # Create Fit_dict for the specific method
        fit_dict = {"Fit": fitdict,
                    "Effects": attdict,
                    "95% CI": cis,
                    "Vectors": Vectors,
                    "WeightV": np.round(b, 3),
                    "Weights": weights_dict
                    }

        # Append fit_dict to the list
        fit_dicts_list.append({method: fit_dict})

    return fit_dicts_list


def pcr(X, y, objective, donor_names):

    # We begin by taking the SVD of the donor matrix
    # in the pre period

    (u, s, v) = np.linalg.svd(X, full_matrices=False)

    # Then, we estimate the universal rank
    # via USVT

    (n1, n2) = X.shape
    ratio = min(n1, n2) / max(n1, n2)
    rank = universal_rank(s, ratio=ratio)

    # Then, we shave off the irrelevant singular values

    s_rank = s[:rank]
    u_rank = u[:, :rank]
    v_rank = v[:rank, :]

    low_rank_component = np.dot(u_rank, np.dot(np.diag(s_rank), v_rank))
    
    weights = Opt.SCopt(n2, y[:X.shape[0]], X.shape[0], low_rank_component, model=objective,  donor_names=donor_names)

    return weights


class Opt:
    @staticmethod
    def SCopt(Nco, y, t1, X, model="MSCb", donor_names=None):

        # Check if matrix dimensions allow multiplication
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of columns in X must be equal to the number of rows in y.")

        if model in ["MSCa", "MSCc"]:
            Nco += 1
            if donor_names:
                donor_names = ["Intercept"] + list(donor_names)

        # Define the optimization variable
        beta = cp.Variable(Nco)

        # Preallocate memory for constraints
        constraints = [beta >= 0] if model != "OLS" else []

        # If intercept added, append the intercept column to X
        if model in ["MSCa", "MSCc"]:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        # Define the objective function
        objective = cp.Minimize(cp.norm(y[:t1] - X @ beta, 2))

        # Define the constraints based on the selected method
        if model == "SIMPLEX":
            constraints.append(cp.sum(beta) == 1)  # Sum constraint for all coefficients
        elif model == "MSCa":
            constraints.append(cp.sum(beta[1:]) == 1)  # Sum constraint for coefficients excluding intercept
        elif model == "MSCc":
            constraints.append(beta[1:] >= 0)

        prob = cp.Problem(objective, constraints)

        # Solve the problem
        result = prob.solve(solver=cp.CLARABEL)

        return beta.value

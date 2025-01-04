from sklearn.decomposition import PCA
from scipy.interpolate import make_interp_spline
from numpy.linalg import inv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from mlsynth.utils.denoiseutils import universal_rank, svt, spectral_rank
from screenot.ScreeNOT import adaptiveHardThresholding
from numpy.linalg import svd


def SVDCluster(X, y, donor_names):
    """
    Clusters donor units based on their SVD embeddings and returns the subset of donors in the same cluster as the treated unit.

    Parameters:
        X (numpy.ndarray): Donor matrix (n_samples x n_features).
        y (numpy.ndarray): Treated unit (1D array of size n_samples).
        donor_names (list): List of donor names.

    Returns:
        X_sub (numpy.ndarray): Subset of the donor matrix corresponding to the treated cluster.
        selected_donor_names (list): Names of donors in the treated cluster.
    """
    # Stack treated unit with donor matrix and perform SVD
    fully = np.hstack((y[:X.shape[0]].reshape(-1, 1), X)).T
    n, p = fully.shape
    k = (min(n, p) // 2) - 1
    print("The compted k is:", k)
    Y0_rank, Topt_gauss, rank = adaptiveHardThresholding(fully, k, strategy='0')
    _, _, u_rank, s_rank, _ = svt(fully)
    U_est, s_est, Vt_est = svd(Y0_rank, full_matrices=False)

    # Compute embeddings for clustering
    u_embeddings = u_rank * s_rank  # Scale rows by singular values
    u_embeddings = U_est[:, :rank]* s_est[:rank]

    # Determine the optimal number of clusters using silhouette scores
    max_clusters = len(donor_names)
    silhouette_scores = [
        silhouette_score(u_embeddings, KMeans(n_clusters=k, random_state=42).fit_predict(u_embeddings))
        for k in range(2, max_clusters + 1)
    ]
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Adjust index for minimum clusters

    # Perform k-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, init='k-means++').fit(u_embeddings)
    clusters = kmeans.labels_

    # Identify the cluster of the treated unit
    treated_cluster = clusters[0]

    # Get donor indices and names for the treated cluster
    selected_indices = np.where(clusters == treated_cluster)[0]
    selected_indices = selected_indices[selected_indices != 0] - 1  # Exclude treated unit
    selected_donor_names = [donor_names[i] for i in selected_indices]

    # Subset the donor matrix
    X_sub = X[:, selected_indices]

    return X_sub, selected_donor_names, selected_indices



def determine_optimal_clusters(X):
    """
    Function to determine the optimal number of clusters using
    Silhouette score.
    """
    silhouette_scores = []
    max_clusters = min(10, X.shape[0])

    for num_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, init='k-means++')  #
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    optimal_clusters = np.argmax(silhouette_scores)+2
    return optimal_clusters


def fpca(X):
    """
    Function to calculating our FPC scores
    """
    # Use X.shape[1] for the number of time points
    x = np.linspace(0, 1, num=X.shape[1])

    # create B-spline basis
    k = 3  # degree of the spline
    # Transpose X to match the required shape
    spl = make_interp_spline(x, X.T, k=k)
    Xfun = spl(x)

    # perform PCA on the functional data
    pca = PCA()
    # Transpose Xfun to match the required shape
    X_pca = pca.fit_transform(Xfun.T)

    U, S, Vt = np.linalg.svd(Xfun.T)

    (n1, n2) = Xfun.T.shape
    ratio = min(n1, n2) / max(n1, n2)

    rank = universal_rank(S,ratio =ratio)
    specrank = spectral_rank(S,t=.90)

    X_pca = X_pca[:, :spectral_rank(S,t=.90)]

    # Scale PCA scores
    cluster_x = (X_pca - np.mean(X_pca, axis=0)) / np.std(X_pca, axis=0)

    # Determine the optimal number of clusters
    optimal_clusters = determine_optimal_clusters(cluster_x)

    return optimal_clusters, cluster_x, spectral_rank(S,t=.90)


def PDAfs(y, donormatrix, t1, t, N):
    """
    Select donor units based on minimizing the Sum of Squared Errors (SSE) and fit a model.

    Parameters:
        y (numpy.ndarray): Outcome variable (length t)
        donormatrix (numpy.ndarray): Donor units matrix (shape: t x N)
        t1 (int): Number of pre-treatment periods
        t (int): Total number of periods (pre + post treatment)
        N (int): Number of donor units

    Returns:
        dict: A dictionary containing the selected donor units, model coefficients, and calculated Q values
    """
    y1_t1 = y[:t1]  # Pre-treatment outcomes for the treated unit
    y0_t1 = donormatrix[:t1]  # Pre-treatment outcomes for donor units
    y1_t2 = y[t1:t]  # Post-treatment outcomes for the treated unit

    R2 = np.zeros(N)  # To store SSE for each donor unit

    # Step 1: Select initial donor unit (based on SSE minimization)
    for j in range(N):
        X = np.column_stack((np.ones(t1), y0_t1[:, j]))
        b = inv(X.T @ X) @ X.T @ y1_t1
        SSE = (X @ b).T @ (X @ b)
        R2[j] = SSE
    select = np.argmax(R2)

    # Step 2: Fit model using selected donor unit
    X = np.column_stack((np.ones(t1), y0_t1[:, select]))
    b = inv(X.T @ X) @ X.T @ y1_t1
    y_hat = np.dot(X, b)
    e = y1_t1 - y_hat
    sigma_sq = np.dot(e.T, e) / t1
    B = np.log(np.log(N)) * np.log(t1) / t1
    Q_new = np.log(sigma_sq) + B
    Q_old = Q_new + 1  # Set initial value to start the while loop

    # Step 3: Iterative process to refine donor unit selection based on SSE
    while Q_new < Q_old:
        left = np.setdiff1d(np.arange(N), select)  # Remaining donor units
        y0_t1_left = y0_t1[:, left]  # Outcomes of the remaining donor units

        Q_old = Q_new  # Update old Q value

        R2 = np.zeros(len(left))  # SSE for remaining donor units

        # Step 4: Calculate SSE for remaining donor units
        for j in range(len(left)):
            X = np.column_stack((np.ones(t1), y0_t1[:, select], y0_t1_left[:, j]))
            b = inv(X.T @ X) @ X.T @ y1_t1
            SSE = (X @ b).T @ (X @ b)
            R2[j] = SSE

        # Step 5: Select the donor unit with the minimum SSE
        index = left[np.argmax(R2)]
        select = np.append(select, index)  # Add selected donor unit to the list

        k = len(select)  # Number of selected donor units

        # Step 6: Refit the model with new set of donor units
        X = np.column_stack((np.ones(t1), y0_t1[:, select]))
        b = inv(X.T @ X) @ X.T @ y1_t1
        y_hat = np.dot(X, b)
        e = y1_t1 - y_hat
        sigma_sq = np.dot(e.T, e) / t1
        B = np.log(np.log(N)) * k * np.log(t1) / t1
        Q_new = np.log(sigma_sq) + B

    # Return results as a dictionary
    return {
        "selected_donors": select,
        "model_coefficients": b,
        "Q_new": Q_new,
        "sigma_sq": sigma_sq,
        "y_hat": y_hat
    }


from sklearn.decomposition import PCA
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

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
    optimal_clusters = np.argmax(silhouette_scores) + 2
    return optimal_clusters


def fpca(X):
    """
    Function to calculating our FPC scores
    """
    # Use X.shape[1] for the number of time points
    x = np.linspace(0, 1, num=X.shape[1])

    # create B-spline basis
    k = 5  # degree of the spline
    # Transpose X to match the required shape
    spl = make_interp_spline(x, X.T, k=k)
    Xfun = spl(x)

    # perform PCA on the functional data
    pca = PCA()
    # Transpose Xfun to match the required shape
    X_pca = pca.fit_transform(Xfun.T)

    U, S, Vt = np.linalg.svd(Xfun.T)

    # Calculate the total variance
    total_variance = np.sum(S ** 2)

    # Set the desired threshold for explained variance (e.g., 95%)
    threshold_explained_variance = 0.95 * total_variance

    # Initialize variables for the cumulative explained variance and the number of singular values
    cumulative_explained_variance = 0
    num_singular_values = 0

    # Loop through the singular values to calculate cumulative explained variance
    for singular_value in S:
        cumulative_explained_variance += singular_value ** 2
        num_singular_values += 1
        if cumulative_explained_variance >= threshold_explained_variance:
            break

    # Now, num_singular_values contains the number of singular values
    # required to explain 95% of the variation.
    # print("Number of singular values explaining 95% of variation:", num_singular_values)


    # Sample list (replace this with your list)
    my_list = [num_singular_values]

    # Find the minimum number in the list
    bestnum = min(my_list)

    X_pca = X_pca[:, :bestnum]

    # Scale PCA scores
    cluster_x = (X_pca - np.mean(X_pca, axis=0)) / np.std(X_pca, axis=0)

    # Determine the optimal number of clusters
    optimal_clusters = determine_optimal_clusters(cluster_x)

    return optimal_clusters, cluster_x, bestnum
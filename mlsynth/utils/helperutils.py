import numpy as np

def prenorm(X, target=100):
    """
    Normalize a vector or matrix so that the last row is equal to `target` (default: 100).

    Parameters:
        X (np.ndarray): Vector (1D) or matrix (2D) to normalize.
        target (float): Value to normalize the last row to (default is 100).

    Returns:
        np.ndarray: Normalized array.
    """
    X = np.asarray(X)

    denom = X[-1] if X.ndim == 1 else X[-1, :]

    # Check for division by zero
    if np.any(denom == 0):
        raise ZeroDivisionError("Division by zero in prenorm function.")

    return X / denom * target

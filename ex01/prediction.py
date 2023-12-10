import numpy as np


def predict_(x, theta):
    """
    Computes the prediction vector y_hat from two non-empty numpy arrays.
    Args:
        x: a numpy.array, a matrix of dimensions m * n
        (m = number of examples, n = number of features).
        theta: a numpy.array, a vector of dimensions (n + 1) * 1
        (including the intercept term).
    Returns:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy arrays,
        or their dimensions are not appropriate,
        or they are not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or theta.size == 0:
        return None
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if theta.shape[0] != x.shape[1] + 1:
        return None

    # Add an intercept term of 1s as the first column of x
    intercept = np.ones((x.shape[0], 1))
    x = np.hstack((intercept, x))

    return x.dot(theta)

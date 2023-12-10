import numpy as np


def gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy arrays,
    without any for-loop. The three arrays must have compatible dimensions.
    Args:
        x: numpy.array, a matrix of dimension m * n.
        y: numpy.array, a vector of dimension m * 1.
        theta: numpy.array, a vector of dimension (n + 1) * 1.
    Returns:
        The gradient as a numpy.array, a vector of dimensions (n + 1) * 1,
        containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.array,
        or if their dimensions are not compatible,
        or if they are not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and isinstance(theta, np.ndarray)
    ):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None

    m = x.shape[0]
    x = np.insert(x, 0, 1, axis=1)

    y_hat = x.dot(theta)
    error = y_hat - y

    return (1 / m) * x.T.dot(error)

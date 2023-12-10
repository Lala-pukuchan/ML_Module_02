import numpy as np


def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not matching.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    # Check if x or theta is not a numpy array or if they are empty
    if (
        not isinstance(x, np.ndarray)
        or not x.size
        or not isinstance(theta, np.ndarray)
        or not theta.size
    ):
        return None

    # Check if the dimensions of x and theta are matching
    if theta.shape[0] != x.shape[1] + 1:
        return None

    # Add a column of ones to x to account for the bias term in theta
    x = np.hstack((np.ones((x.shape[0], 1)), x))

    # Compute the prediction using dot product
    y_hat = np.dot(x, theta)

    return y_hat

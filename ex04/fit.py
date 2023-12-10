import numpy as np


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
            (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
            (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
            (number of features + 1, 1).
        alpha: has to be a float, the learning rate.
        max_iter: has to be an int, the number of iterations done during the gradient descent.
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or not isinstance(alpha, float)
        or not isinstance(max_iter, int)
    ):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        return None
    if x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None
    if max_iter < 0:
        return None

    x = np.insert(x, 0, 1, axis=1)

    for i in range(max_iter):
        m = x.shape[0]
        error = np.dot(x, theta) - y
        theta = theta - alpha * (np.dot(x.T, error) / m)

    return theta

import numpy as np


def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop. The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Return:
        The mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of expected type.
        Raises:
        This function should not raise any Exception.
    """
    if y.size == 0 or y_hat.size == 0 or y.size != y_hat.size:
        return None

    # Flatten the arrays
    y = y.flatten()
    y_hat = y_hat.flatten()

    return np.dot((y_hat - y), (y_hat - y)) / (2 * y.size)

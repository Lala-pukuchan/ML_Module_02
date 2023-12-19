import numpy as np
from loss import loss_

X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
print("---ex.1---")
# Example 1:
print("loss_: \n", loss_(X, Y))
# Output:
print("expected: \n", 2.142857142857143)
# 2.142857142857143
print("\n---ex.2---")
# Example 2:
print("loss_: \n", loss_(X, X))
print("expected: \n", 0.0)
# Output:
# 0.0

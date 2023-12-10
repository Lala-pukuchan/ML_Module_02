import numpy as np
from mylinearregression import MyLinearRegression as MyLR

X = np.array([[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [34.0, 55.0, 89.0, 144.0]])
Y = np.array([[23.0], [48.0], [218.0]])
mylr = MyLR([[1.0], [1.0], [1.0], [1.0], [1]])

print("--- Example 0 ---")
y_hat = mylr.predict_(X)
print("my y_hat: \n", y_hat)
print("expected y_hat: \n", np.array([[8.0], [48.0], [323.0]]))

print("--- Example 1 ---")
print("my loss_elem: \n", mylr.loss_elem_(Y, y_hat))
print("expected loss_elem: \n", np.array([[225.0], [0.0], [11025.0]]))

print("--- Example 2 ---")
print("my loss: \n", mylr.loss_(Y, y_hat))
print("expected loss: \n", 1875.0)

print("--- Example 3 ---")
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
print("my fit: \n", mylr.fit_(X, Y))
print("expected fit: \n", np.array([[18.188], [2.767], [-0.374], [1.392], [0.017]]))

print("--- Example 4 ---")
y_hat = mylr.predict_(X)
print("my y_hat: \n", y_hat)
print("expected y_hat: \n", np.array([[23.417], [47.489], [218.065]]))

print("--- Example 5 ---")
print("my loss_elem: \n", mylr.loss_elem_(Y, y_hat))
print("expected loss_elem: \n", np.array([[0.174], [0.26], [0.004]]))

print("--- Example 6 ---")
print("my loss: \n", mylr.loss_(Y, y_hat))
print("expected loss: \n", 0.0732)
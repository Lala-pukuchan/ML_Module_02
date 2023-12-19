import numpy as np
from fit import fit_
from ex01.prediction import predict_

x = np.array([[0.2, 2.0, 20.0], [0.4, 4.0, 40.0], [0.6, 6.0, 60.0], [0.8, 8.0, 80.0]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.0], [1.0], [1.0], [1.0]])
print("---ex.1---")
# Example 0:
theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
print(theta2)
# Output:
# array([[41.99..],[0.97..], [0.77..], [-1.20..]])
print("expected: \narray([[41.99..],[0.97..], [0.77..], [-1.20..]])")

print("\n---ex.2---")
# Example 1:
print(predict_(x, theta2))
# Output:
# array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
print("expected: \narray([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])")

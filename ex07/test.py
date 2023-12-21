import numpy as np
import matplotlib.pyplot as plt
from ex07.polynomial_model import add_polynomial_features
from ex05.mylinearregression import MyLinearRegression as MyLR

x = np.arange(1, 6).reshape(-1, 1)
print("---Example 0---")
# Example 0:
print("my polynominal: \n", add_polynomial_features(x, 3))
# Output:
print(
    "expected polynominal: \n",
    np.array([[1, 1, 1], [2, 4, 8], [3, 9, 27], [4, 16, 64], [5, 25, 125]]),
)
print("\n---Example 1---")
# Example 1:
print("my polynominal: \n", add_polynomial_features(x, 6))
# Output:
print(
    "expected polynominal: \n",
    np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [2, 4, 8, 16, 32, 64],
            [3, 9, 27, 81, 243, 729],
            [4, 16, 64, 256, 1024, 4096],
            [5, 25, 125, 625, 3125, 15625],
        ]
    ),
)

x = np.arange(1,11).reshape(-1,1)
y = np.array([[ 1.39270298],
[ 3.88237651],
[ 4.37726357],
[ 4.63389049],
[ 7.79814439],
[ 6.41717461],
[ 8.63429886],
[ 8.19939795],
[10.37567392],
[10.68238222]])
plt.scatter(x,y)
plt.savefig("results/ex07/result_ex07_figure-1.png")
plt.close()

# Build the model:
x_ = add_polynomial_features(x, 3)
theta4 = np.array([[1], [1], [1], [1]]).reshape(-1, 1)
my_lr = MyLR(theta4, alpha=1e-6, max_iter=600000)
print("theta4 before:", theta4)
my_lr.fit_(x_, y)
print("theta4 after:", my_lr.theta)
# Plot:
## To get a smooth curve, we need a lot of data points
continuous_x = np.arange(1,10.01, 0.01).reshape(-1,1)
continuous_x_ = add_polynomial_features(continuous_x, 3)
y_hat = my_lr.predict_(continuous_x_)
plt.scatter(x,y)
plt.plot(continuous_x, y_hat, color='orange')
plt.savefig("results/ex07/result_ex07_figure-2.png")
plt.close()

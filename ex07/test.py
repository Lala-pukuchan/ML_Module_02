import numpy as np
from ex07.polynomial_model import add_polynomial_features

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

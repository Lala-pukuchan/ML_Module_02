import numpy as np
from ex09.data_spliter import data_spliter

x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

print("\nExample 1:")
# Example 1:
print("my data spliter\n", data_spliter(x1, y, 0.8))

print("\nExample 2:")
# Example 2:
print("my data spliter\n", data_spliter(x1, y, 0.5))
x2 = np.array([[1, 42], [300, 10], [59, 1], [300, 59], [10, 42]])
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

print("\nExample 3:")
# Example 3:
print("my data spliter\n", data_spliter(x2, y, 0.8))

print("\nExample 4:")
# Example 4:
print("my data spliter\n", data_spliter(x2, y, 0.5))

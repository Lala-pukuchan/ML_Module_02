import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from ex05.mylinearregression import MyLinearRegression as MyLR
from ex09.data_spliter import data_spliter
from ex07.polynomial_model import add_polynomial_features

# Load data
data = pd.read_csv("ex10/space_avocado.csv")
x = np.array(data[["weight", "prod_distance", "time_delivery"]])
y = np.array(data["target"]).reshape(-1, 1)

# Split data
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

# Store performance of models
models_performance = {}

# Train and evaluate models with polynomial features up to degree 4
for degree in range(1, 5):

    # Create polynomial features
    x_train_poly = add_polynomial_features(x_train, degree)
    x_test_poly = add_polynomial_features(x_test, degree)

    # Initialize the model
    theta = np.random.randn(x_train_poly.shape[1] + 1, 1)
    if degree == 1:
        alpha = 1e-7
        max_iter = 500000
    elif degree == 2:
        alpha = 1e-15
        max_iter = 1000000
    elif degree == 3:
        alpha = 1e-20
        max_iter = 10000000
    elif degree == 4:
        alpha = 1e-27
        max_iter = 10000000
    model = MyLR(theta, alpha, max_iter)

    # Fit the model
    model.fit_(x_train_poly, y_train)

    # Evaluate the model
    train_mse = model.mse_(y_train, model.predict_(x_train_poly))
    test_mse = model.mse_(y_test, model.predict_(x_test_poly))
    
    print("--- degree", degree, "---")
    print("train_mse:", train_mse)
    print("test_mse :", test_mse)
    print("theta :", model.theta)

    # Store model and performance
    models_performance[degree] = {"model": model, "train_mse": train_mse, "test_mse": test_mse}

# Save the models to a file
with open("ex10/models.pkl", "wb") as file:
    pickle.dump(models_performance, file)

# Plotting MSE for each model
degrees = list(models_performance.keys())
test_mses = [models_performance[d]["test_mse"] for d in degrees]

plt.figure()
plt.plot(degrees, test_mses, marker='o', color='b', label='Test MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('Model Evaluation - Test MSE vs Polynomial Degree')
plt.legend()
plt.savefig("results/ex10/result_ex10_figure-1.png")
plt.close()

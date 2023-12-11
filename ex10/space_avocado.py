import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ex05.mylinearregression import MyLinearRegression as MyLR
from ex07.polynomial_model import add_polynomial_features

# Load dataset
data = pd.read_csv("ex10/space_avocado.csv")
x = np.array(data[["weight", "prod_distance", "time_delivery"]])
y = np.array(data["target"]).reshape(-1, 1)

# Load models and scores from the file
with open("ex10/models.pkl", "rb") as file:
    models = pickle.load(file)

# Find the best model
best_degree = min(models, key=lambda k: models[k]["mse"])
best_theta = models[best_degree]["theta"]

# Create polynomial features
x_poly = add_polynomial_features(x, best_degree)
best_model = MyLR(best_theta, alpha=1e-5, max_iter=600000)
predictions = best_model.predict_(x_poly)

# Plotting actual vs predicted values for each feature
feature_names = ["weight", "prod_distance", "time_delivery"]
plt.figure(figsize=(18, 6))
for i in range(x.shape[1]):
    plt.subplot(1, 3, i + 1)
    plt.scatter(x[:, i], y, color="blue", label="Actual")
    plt.scatter(x[:, i], predictions, color="red", label="Predicted", marker="x")
    plt.title(f"Feature: {feature_names[i]}")
    plt.xlabel(feature_names[i])
    plt.ylabel("Target")
    plt.legend()
plt.suptitle(f"Actual vs Predicted - Best Model (Degree {best_degree})")
plt.savefig("results/ex10/result_ex10_figure-1.png")
plt.close()

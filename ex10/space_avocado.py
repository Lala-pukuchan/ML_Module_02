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

# Identify the best model (minimum test MSE)
best_degree = min(models, key=lambda k: models[k]["test_mse"])
best_model_info = models[best_degree]
best_model = best_model_info["model"]
x_test_poly = best_model_info["x_test_poly"]
y_test = best_model_info["y_test"]

# Predict using the best model
predictions = best_model.predict_(x_test_poly)

# Plot actual and predict
y_pred = best_model.predict_(x_test_poly)

# Plot 1: Feature 1 (Weight) and Feature 2 (Production Distance)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection="3d")
ax1.scatter(
    x_test_poly[:, 0], x_test_poly[:, 1], y_test[:, 0], c="b", label="Actual Price"
)
ax1.scatter(
    x_test_poly[:, 0], x_test_poly[:, 1], y_pred[:, 0], c="r", label="Predicted Price"
)
ax1.set_xlabel("Weight")
ax1.set_ylabel("Production Distance")
ax1.set_zlabel("Price")
ax1.legend()
plt.title("Price Prediction with Weight and Production Distance")
plt.savefig("results/ex10/result_ex10_figure-1.png")
plt.close()

# Plot 2: Feature 1 (Weight) and Feature 3 (Time Delivery)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection="3d")
ax2.scatter(
    x_test_poly[:, 0], x_test_poly[:, 2], y_test[:, 0], c="b", label="Actual Price"
)
ax2.scatter(
    x_test_poly[:, 0], x_test_poly[:, 2], y_pred[:, 0], c="r", label="Predicted Price"
)
ax2.set_xlabel("Weight")
ax2.set_ylabel("Time Delivery")
ax2.set_zlabel("Price")
ax2.legend()
plt.title("Price Prediction with Weight and Time Delivery")
plt.savefig("results/ex10/result_ex10_figure-2.png")
plt.close()

# Plot 3: Feature 2 (Production Distance) and Feature 3 (Time Delivery)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection="3d")
ax3.scatter(
    x_test_poly[:, 1], x_test_poly[:, 2], y_test[:, 0], c="b", label="Actual Price"
)
ax3.scatter(
    x_test_poly[:, 1], x_test_poly[:, 2], y_pred[:, 0], c="r", label="Predicted Price"
)
ax3.set_xlabel("Production Distance")
ax3.set_ylabel("Time Delivery")
ax3.set_zlabel("Price")
ax3.legend()
plt.title("Price Prediction with Production Distance and Time Delivery")
plt.savefig("results/ex10/result_ex10_figure-3.png")
plt.close()

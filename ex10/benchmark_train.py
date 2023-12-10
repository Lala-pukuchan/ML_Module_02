import numpy as np
import pandas as pd
import pickle
from ex05.mylinearregression import MyLinearRegression as MyLR
from ex09.data_spliter import data_spliter
from ex07.polynomial_model import add_polynomial_features


# Load data
data = pd.read_csv("ex10/space_avocado.csv")
x = np.array(data[["weight", "prod_distance", "time_delivery"]])
y = np.array(data["target"]).reshape(-1, 1)

# Split data
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

# Reshape y_train and y_test if they are not two-dimensional
if y_train.ndim == 1:
    y_train = y_train.reshape(-1, 1)
if y_test.ndim == 1:
    y_test = y_test.reshape(-1, 1)

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)

# Train models
models = {}

# Train models with polynomial features of degrees 1 to 4
for degree in range(1, 5):
    # Create polynomial features
    x_train_poly = add_polynomial_features(x_train, degree)

    # Initialize model with zeros for theta
    theta = np.zeros((x_train_poly.shape[1], 1))
    model = MyLR(theta)

    # Train model
    model.fit_(x_train_poly, y_train)

    # Evaluate model
    x_test_poly = add_polynomial_features(x_test, degree)
    score = model.mse_(x_test_poly, y_test)

    # Save model and score
    models[degree] = {"theta": model.theta, "mse": score}

# Save models and scores to a file
with open("ex10/models_and_scores.pkl", "wb") as file:
    pickle.dump(models, file)

print("Models and scores saved successfully.")

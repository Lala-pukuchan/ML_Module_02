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

# Train models
models_dict = {}

# Train models with polynomial features of degrees 1 to 4
for degree in range(1, 6):
    # Create polynomial features
    x_train_poly = add_polynomial_features(x_train, degree)

    # number of features is the number of columns of x_train_poly + 1
    num_features = x_train_poly.shape[1] + 1
    theta = np.random.randn(num_features, 1)
    model = MyLR(theta, alpha=1e-7, max_iter=1000000)
    # [CHECK] can't converge for degree 2 and above

    # Train model
    model.fit_(x_train_poly, y_train)
    predictions = model.predict_(x_train_poly)
    mse = model.mse_(y_train, predictions)
    print(f"MSE for degree {degree}: {mse}\nthetas: {model.theta}\n")

    # Store model and its MSE
    models_dict[degree] = {"theta": model.theta, "mse": mse}

# Save models and MSE scores to a Pickle file
with open("ex10/models.pkl", "wb") as file:
    pickle.dump(models_dict, file)

print("Models and MSE scores saved successfully.")

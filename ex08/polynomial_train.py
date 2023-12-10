import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ex05.mylinearregression import MyLinearRegression as MyLR

# Load the data
data = pd.read_csv("ex08/are_blue_pills_magics.csv")
X = np.array(data["Micrograms"]).reshape(-1, 1)
Y = np.array(data["Score"]).reshape(-1, 1)


# Function to add polynomial features
def add_polynomial_features(x, power):
    return np.hstack([x**i for i in range(1, power + 1)])


# Initialize theta for higher degree models
theta4 = np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1)
theta5 = np.array([[1140], [-1850], [1110], [-305], [40], [-2]]).reshape(-1, 1)
theta6 = np.array([[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]]).reshape(
    -1, 1
)

# Training models and calculating MSE
mse_scores = []
models = []
x_model = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

for degree in range(1, 7):
    X_poly = add_polynomial_features(X, degree)

    if degree == 4:
        my_lreg = MyLR(theta4, alpha=1e-6, max_iter=600000)
    elif degree == 5:
        my_lreg = MyLR(theta5, alpha=1e-8, max_iter=600000)
    elif degree == 6:
        my_lreg = MyLR(theta6, alpha=1e-9, max_iter=600000)
    else:
        my_lreg = MyLR(np.ones((degree + 1, 1)), alpha=1e-5, max_iter=600000)

    my_lreg.fit_(X_poly, Y)
    predictions = my_lreg.predict_(X_poly)
    mse = my_lreg.mse_(Y, predictions)
    mse_scores.append(mse)
    models.append(my_lreg)
    print(f"MSE for degree {degree}: {mse}")

# Plotting the bar plot for MSEs
plt.figure()
plt.bar(range(1, 7), mse_scores, color="blue")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("MSE of Polynomial Models")
plt.savefig("results/ex08/result_ex08_figure-1.png")
plt.close()

# Plotting the models and data points
plt.figure()
plt.scatter(X, Y, color="green", label="Actual Scores")
for degree, model in enumerate(models, start=1):
    X_poly_model = add_polynomial_features(x_model, degree)
    y_model = model.predict_(X_poly_model)
    plt.plot(x_model, y_model, label=f"Degree {degree}")

plt.xlabel("Micrograms")
plt.ylabel("Score")
plt.title("Polynomial Models and Actual Data")
plt.legend()
plt.savefig("results/ex08/result_ex08_figure-2.png")
plt.close()

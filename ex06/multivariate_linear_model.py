import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ex05.mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("ex06/spacecraft_data.csv")

print("--- Part.1 ---")
X1 = np.array(data[["Age"]])
Y = np.array(data[["Sell_price"]])

myLR_age = MyLR(theta=[[1000.0], [-1.0]], alpha=2.5e-5, max_iter=100000)
myLR_age.fit_(X1[:, 0].reshape(-1, 1), Y)
y_pred = myLR_age.predict_(X1[:, 0].reshape(-1, 1))

myLR_age.mse_(y_pred, Y)
print("my mse   :", myLR_age.mse_(y_pred, Y))
print("expected :", 55736.86719)

plt.grid(True, which="both", linestyle="-", linewidth=0.5)
plt.xlabel("x1: age (in years)")
plt.ylabel("y: sell price (in keuros)")
plt.scatter(X1, y_pred, color="cyan", s=5, label="Predicted sell price")
plt.scatter(X1, Y, color="blue", label="Sell price")
plt.legend()
plt.savefig("results/ex06/result_ex06_figure1-1.png")
plt.close()

# initial theta, alpha and max_iter values should be changed to get a better fit
X2 = np.array(data[["Thrust_power"]])
myLR_thrust = MyLR(theta=[[4.0], [1.0]], alpha=1e-4, max_iter=100000)
myLR_thrust.fit_(X2[:, 0].reshape(-1, 1), Y)
y_pred = myLR_thrust.predict_(X2[:, 0].reshape(-1, 1))

plt.grid(True, which="both", linestyle="-", linewidth=0.5)
plt.xlabel("x2: thrust power (in 10Km/s)")
plt.ylabel("y: sell price (in keuros)")
plt.scatter(X2, y_pred, color="lime", s=5, label="Predicted sell price")
plt.scatter(X2, Y, color="green", label="Sell price")
plt.legend()
plt.savefig("results/ex06/result_ex06_figure1-2.png")
plt.close()

X3 = np.array(data[["Terameters"]])
myLR_thrust = MyLR(theta=[[1.0], [1.0]], alpha=2e-4, max_iter=100000)
myLR_thrust.fit_(X3[:, 0].reshape(-1, 1), Y)
y_pred = myLR_thrust.predict_(X3[:, 0].reshape(-1, 1))

plt.grid(True, which="both", linestyle="-", linewidth=0.5)
plt.xlabel("x3: distance totalizer value of spacecraft (in Tmeter)")
plt.ylabel("y: sell price (in keuros)")
plt.scatter(X3, y_pred, color="pink", s=5, label="Predicted sell price")
plt.scatter(X3, Y, color="purple", label="Sell price")
plt.legend()
plt.savefig("results/ex06/result_ex06_figure1-3.png")
plt.close()

print("\n--- Part.2 ---")
X = np.array(data[["Age", "Thrust_power", "Terameters"]])
Y = np.array(data[["Sell_price"]])
# my_lreg = MyLR(theta=[1.0, 1.0, 1.0, 1.0], alpha=1e-4, max_iter=600000)
my_lreg = MyLR(theta=[1.0, 1.0, 1.0, 1.0], alpha=1e-5, max_iter=600000)

initial_prediction = my_lreg.predict_(X)
print("my mse   :", my_lreg.mse_(Y, initial_prediction))
print("expected :", 144044.877)

my_lreg.fit_(X, Y)
print("my theta :", my_lreg.theta)
print("expected :", np.array([[334.994], [-22.535], [5.857], [-2.586]]))

prediction_after_fit = my_lreg.predict_(X)
print("my mse after fit :", my_lreg.mse_(Y, prediction_after_fit))
print("expected         :", 586.896999)

# Plotting
features = ["Age", "Thrust_power", "Terameters"]
for i, feature in enumerate(features):
    plt.figure(i)
    plt.scatter(data[feature], Y, color="blue", label="Actual Prices")
    plt.scatter(
        data[feature],
        prediction_after_fit,
        color="red",
        label="Predicted Prices",
        marker="x",
        s=5,
    )
    plt.xlabel(feature)
    plt.ylabel("Sell Price")
    plt.title(f"Actual vs Predicted Sell Prices vs {feature}")
    plt.legend()
    plt.savefig(f"results/ex06/result_ex06_figure2-{i}.png")
    plt.close()

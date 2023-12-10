# Pseudocode - replace with actual implementation
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR

# Load the best model parameters from file
# ...

# Load dataset
data = pd.read_csv("space_avocado.csv")
# Split data into training and test sets
# ...

# Train the best model on the training set
best_model = MyLR(...)
best_model.fit_(X_train_poly, Y_train)

# Evaluate and plot
# ...

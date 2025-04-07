# generate_models.py

import os
import pickle
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

# Create ml_models directory if it doesn't exist
if not os.path.exists("ml_models"):
    os.makedirs("ml_models")

# ----- Classification Model: Logistic Regression on Iris Dataset -----
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)
with open("ml_models/logistic_model.pkl", "wb") as f:
    pickle.dump(logistic_model, f)
print("Saved logistic_model.pkl")

# ----- Regression Model: Linear Regression on Diabetes Dataset -----
diabetes = load_diabetes()
X_diabetes, y_diabetes = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
with open("ml_models/linear_regression_model.pkl", "wb") as f:
    pickle.dump(linear_model, f)
print("Saved linear_regression_model.pkl")

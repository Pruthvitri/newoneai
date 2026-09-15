import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 45, 50, 55, 60, 65, 72, 80, 88]
}

df = pd.DataFrame(data)

# Display dataset
print("Dataset:")
print(df)

# Check information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Select features and target
X = df[["Study_Hours"]]
y = df["Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Results")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# New prediction
new_student = [[7.5]]
prediction = model.predict(new_student)

print("\nPrediction")
print("Study Hours:", new_student[0][0])
print("Predicted Marks:", prediction[0])

# Visualization
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Study_Hours"],
    df["Marks"],
    label="Actual Data"
)

plt.plot(
    df["Study_Hours"],
    model.predict(df[["Study_Hours"]]),
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction using Linear Regression")

plt.legend()
plt.show()

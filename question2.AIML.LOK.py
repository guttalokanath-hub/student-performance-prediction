import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = {
    "study_hours": [1,2,3,4,5,6,7,8,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,3,4,5,6,7,8,9,10,2,3,4,5,6,7,8,9,10,5],
    "attendance":  [60,65,70,75,80,85,90,95,68,72,78,82,88,91,94,96,62,66,71,76,81,86,89,93,97,69,73,77,83,87,92,95,74,79,84,88,91,94,98,99,67,70,75,80,85,90,92,96,98,88],
    "performance_score": [40,45,50,55,60,65,70,75,48,52,58,62,68,72,76,80,42,46,51,56,61,66,69,74,78,49,53,57,63,67,71,75,54,59,64,68,72,76,81,85,47,50,55,60,65,70,73,78,82,66]
}
df = pd.DataFrame(data)
print("Dataset Preview:\n", df.head())
X = df[["study_hours", "attendance"]]
y = df["performance_score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 2))
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print("\nPredictions vs Actual:")
print(results.head())
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Performance")
plt.show()
plt.figure()
plt.scatter(df["study_hours"], df["performance_score"])
plt.xlabel("Study Hours")
plt.ylabel("Performance Score")
plt.title("Study Hours vs Performance")
plt.show()
plt.figure()
plt.scatter(df["attendance"], df["performance_score"])
plt.xlabel("Attendance")
plt.ylabel("Performance Score")
plt.title("Attendance vs Performance")
plt.show()

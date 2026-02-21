import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt   # 👈 ADD THIS

# Sample dataset
data = {
    'area': [1500, 2000, 2500, 1800, 2200, 1200, 3000, 1600, 2800, 2100],
    'bedrooms': [3, 4, 4, 3, 4, 2, 5, 3, 5, 4],
    'bathrooms': [2, 3, 3, 2, 3, 1, 4, 2, 4, 3],
    'age': [10, 5, 8, 15, 3, 20, 2, 12, 6, 7],
    'price': [300000, 400000, 500000, 350000, 450000, 200000, 600000, 320000, 580000, 420000]
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# =========================
# 📊 VISUALIZATIONS
# =========================

# 1️⃣ Actual vs Predicted Prices
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# 2️⃣ Residual Plot
residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()

# Save the model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
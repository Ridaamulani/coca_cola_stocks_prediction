import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Load cleaned dataset
df = pd.read_csv("cleaned_coca_cola_stock.csv")

# Check for missing values
if df.isnull().sum().sum() > 0:
    print("Warning: Dataset contains missing values. Consider handling them.")

# Define Features & Target Variable
features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return']
target = 'Close'

X = df[features]
y = df[target]

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into Training & Testing Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

# Train the Model with Hyperparameter Tuning
model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

# Predict & Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Feature Importance Analysis
feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:\n", feature_importance)

# Save the Model & Scaler
joblib.dump(model, "coca_cola_model.pkl")
joblib.dump(scaler, "scaler.pkl")


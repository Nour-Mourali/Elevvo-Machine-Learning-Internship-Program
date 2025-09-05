# Walmart Sales Forecasting - Regression with Time Features
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. Load Data
import kagglehub
from pathlib import Path
import os
import shutil

# Create a data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Download dataset from Kaggle
download_path = Path(kagglehub.dataset_download("aslanahmedov/walmart-sales-forecast"))
print(f"Download path: {download_path}")

# Copy all files to our data directory
for file_path in download_path.rglob("*.*"):
    if file_path.is_file():
        shutil.copy2(file_path, data_dir)
        print(f"Copied: {file_path.name}")

# Try to find the CSV file in our data directory
csv_files = list(data_dir.glob("*.csv"))
if csv_files:
    df = pd.read_csv(csv_files[0])
else:
    raise FileNotFoundError("Could not find Walmart_Store_sales.csv in the data directory")

# Load and merge the data
train_df = pd.read_csv(data_dir / "train.csv")
features_df = pd.read_csv(data_dir / "features.csv")
stores_df = pd.read_csv(data_dir / "stores.csv")

# Merge the dataframes
df = train_df.merge(features_df, on=['Store', 'Date'], how='left')
df = df.merge(stores_df, on=['Store'], how='left')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date'])

# Print dataset info for verification
print("\nDataset Info:")
print(df.info())
print("\nSample of merged data:")
print(df.head())

# 2. Seasonal Decomposition (example on one store)
example_series = df[df['Store']==1].set_index('Date')['Weekly_Sales']
decomp = seasonal_decompose(example_series, model='additive', period=52)  # weekly data, yearly seasonality

decomp.plot()
plt.show()

# 3. Feature Engineering

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

# Lag & rolling stats
df['lag_1'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1)
df['lag_2'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(2)
df['lag_4'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(4)
df['rolling_mean_4'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1).rolling(4).mean()
df['rolling_mean_12'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1).rolling(12).mean()

# Drop NA rows
df = df.dropna()


# 4. Prepare Data

features = ['Store','Dept','year','month','weekofyear',
            'lag_1','lag_2','lag_4','rolling_mean_4','rolling_mean_12']
X = df[features]
y = df['Weekly_Sales']

# 5. Time-Aware Validation

tscv = TimeSeriesSplit(n_splits=5)

models = {
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=-1, random_state=42)
}

results = {}

for name, model in models.items():
    fold = 1
    mae_scores, rmse_scores = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"{name} | Fold {fold} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        fold += 1

    results[name] = {"MAE": np.mean(mae_scores), "RMSE": np.mean(rmse_scores)}

# 6. Visualization and Model Comparison
plt.figure(figsize=(20,10))

# Plot actual vs predicted for both models
plt.subplot(2,1,1)
plt.plot(y_test.values, label="Actual", color="black", linewidth=2, alpha=0.7)
for name, model in models.items():
    y_pred = model.predict(X_test)
    plt.plot(y_pred, label=f"{name}", 
             color="blue" if name == "XGBoost" else "red", 
             alpha=0.6)
plt.title("Actual vs Predicted Sales Comparison (Last Fold)")
plt.xlabel("Time")
plt.ylabel("Weekly Sales")
plt.legend()

# Plot prediction errors
plt.subplot(2,1,2)
for name, model in models.items():
    y_pred = model.predict(X_test)
    errors = y_test.values - y_pred
    plt.plot(errors, label=f"{name} Error", 
             color="blue" if name == "XGBoost" else "red", 
             alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Prediction Errors Over Time")
plt.xlabel("Time")
plt.ylabel("Error (Actual - Predicted)")
plt.legend()

plt.tight_layout()
plt.show()

# Print detailed performance comparison
print("\nDetailed Model Comparison:")
print("-" * 50)
best_mae = float('inf')
best_model = None

for name, res in results.items():
    print(f"\n{name} Performance Metrics:")
    print(f"MAE: {res['MAE']:,.2f}")
    print(f"RMSE: {res['RMSE']:,.2f}")
    
    if res['MAE'] < best_mae:
        best_mae = res['MAE']
        best_model = name

print(f"\n{'-' * 50}")
print(f"Best Performing Model: {best_model}")
print(f"Best MAE Score: {best_mae:,.2f}")

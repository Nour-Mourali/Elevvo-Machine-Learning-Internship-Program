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

df = pd.read_csv("walmart.csv")   # replace with your dataset
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store','Dept','Date'])


# 2. Seasonal Decomposition (example on one store/department)

example_series = df[df['Store']==1][df['Dept']==1].set_index('Date')['Weekly_Sales']
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

# 6. Visualization (last fold example)

plt.figure(figsize=(15,6))
plt.plot(y_test.values, label="Actual", color="black")
plt.plot(y_pred, label=f"{name} Predicted", color="blue")
plt.title("Actual vs Predicted Sales (Last Fold)")
plt.xlabel("Time")
plt.ylabel("Weekly Sales")
plt.legend()
plt.show()

print("\nAverage CV Results:")
for name, res in results.items():
    print(f"{name} -> MAE: {res['MAE']:.2f}, RMSE: {res['RMSE']:.2f}")

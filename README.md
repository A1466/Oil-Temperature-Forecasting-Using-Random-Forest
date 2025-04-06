# 🛠️ Oil Temperature Forecasting Using Random Forest

This project focuses on **forecasting the Oil Temperature (OT)** of electricity transformers using historical data and machine learning techniques. The main objective is to **predict the OT for the next 24 hours** and **visualize it alongside historical data**, enabling early detection of anomalies and efficient maintenance scheduling.

---

## 📌 Project Objective

- Analyze and preprocess historical OT data.
- Build a robust forecasting model using **Random Forest Regression**.
- Extract time-based features and engineer lag/rolling statistics.
- Predict the oil temperature for the **next 24 hours at 1-hour intervals**.
- Visualize:
  - 24-hour OT forecast
  - Actual vs Predicted on validation set
  - Raw OT historical data (with clean datetime x-axis)

---

## 📚 Dependencies / Libraries Used

Make sure to install the following libraries before running the project:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates



🔍 Data Preprocessing
Datetime Parsing: Converted date column to datetime.

Missing Values: Handled using linear interpolation.

Feature Engineering:

Extracted time-based features: hour, dayofweek, month, is_weekend

Created lag features: OT_lag_1, OT_lag_2, OT_lag_3, OT_lag_24

Computed rolling statistics: mean and std over 3 and 24 hours

# 🔥 Oil Temperature Forecasting using Random Forest

This project focuses on **forecasting the Oil Temperature (OT)** of electricity transformers for the next 24 hours using **Random Forest Regression**. Accurate predictions enable early maintenance, preventing potential failures and ensuring operational efficiency.

---

## 📌 Objective

- Build a robust machine learning pipeline using Random Forest.
- Use historical OT and sensor data to forecast oil temperature.
- Predict OT values for the next 24 hours with 1-hour resolution.
- Evaluate the model using both validation and test datasets.
- Visualize model performance with clear plots.

---

## 📁 Dataset

The dataset contains the following files:

- `train.csv` — historical OT data used for training and validation.
- `test.csv` — future OT data used for evaluating test performance.

### Features:
- `date`: Timestamp (hourly frequency)
- Sensor readings: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`
- `OT`: Oil Temperature (target variable)

---

## 📦 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
---
## 🧹 Data Preprocessing

### 🗓 Datetime Handling
- Converted `date` column to datetime format.
- Set `date` as index to perform time-based interpolation.
- Handled missing values using **linear interpolation**.
- Reset index after filling nulls.

---

## 🕒 Time-Based Features
Extracted features from the `date` column to help the model understand time-based patterns:
- `hour`
- `dayofweek`
- `month`
- `is_weekend` (True if Saturday or Sunday)

---

## 🧠 Feature Engineering
To capture temporal patterns and trends in the data:
- **Lag features**:
  - `OT_lag_1`, `OT_lag_2`, `OT_lag_3`, `OT_lag_24`

- **Rolling statistics**:
  - Mean and Standard Deviation over time windows:
    - `OT_roll_mean_3`, `OT_roll_std_3`
    - `OT_roll_mean_24`, `OT_roll_std_24`

---

## 🔄 Feature Scaling
- Used `StandardScaler` to normalize the feature set.
- This ensures that the model treats all input features equally.

---

## 🧠 Model Training
- **Model**: `RandomForestRegressor`  
  - Parameters: `n_estimators=100`, `random_state=42`
- **Split**:  
  - 80% training  
  - 20% validation
- Trained the model on **scaled features and target variable (`OT`)**.

---

## 📊 Evaluation Metrics

Model performance was evaluated on both **validation** and **test** datasets using the following metrics:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

## 🚀 How to Run

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/oil-temperature-forecasting.git


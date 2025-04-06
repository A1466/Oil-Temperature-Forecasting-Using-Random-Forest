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



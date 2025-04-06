# 🔥 Oil Temperature Forecasting using Random Forest

This project is focused on **forecasting the Oil Temperature (OT)** of electricity transformers for the next 24 hours. The goal is to develop a machine learning solution that predicts future OT values using historical data with **Random Forest Regression**, enabling proactive monitoring and maintenance.

---

## 📌 Objective

- Use historical OT data to train a predictive model.
- Forecast OT values for the next 24 hours at 1-hour intervals.
- Evaluate model performance using validation data.
- Visualize predictions and trends effectively.
- Prepare clean and structured code ready for deployment or further enhancements.

---

## 📁 Dataset

The dataset consists of two CSV files:

- `train.csv`: Historical OT data (used for training and validation)
- `test.csv`: Data for future OT values (optional in this pipeline)

Each row includes:

- `date`: Timestamp (in hourly frequency)
- `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`: Sensor readings
- `OT`: Oil Temperature (Target variable)

---

## 📦 Libraries Used

Ensure the following Python libraries are installed:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates


## 🧹 Data Preprocessing

### 🕒 Datetime Handling:
- Parsed `date` column as datetime.
- Interpolated missing values using linear method.
- Extracted time features: `hour`, `day of week`, `month`, `is_weekend`.

### 🛠️ Feature Engineering:
- Created lag features: `OT_lag_1`, `OT_lag_2`, `OT_lag_3`, `OT_lag_24`.
- Created rolling statistics:
  - Mean and standard deviation over 3-hour window.
  - Mean and standard deviation over 24-hour window.

### ⚖️ Scaling:
- Used `StandardScaler` to scale the features.

---

## 🧠 Model Training

- Model: `RandomForestRegressor` with 100 estimators.
- Train-validation split: 80% training, 20% validation.
- Trained on scaled features (`X`) and target (`y`).

---

## 📊 Evaluation Metrics

Model was evaluated on the validation set using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

### ✅ Results:
      Validation MAE: 0.05
      Validation RMSE: 0.09
      R² Score: 1.00
## How to Run
  1 Clone this repository.
  
  2 Place train.csv and test.csv in the same directory as the notebook.
  
  3 Run the notebook main_forecast.ipynb or .py file step-by-step.
  
  4 View the results and generated graphs.

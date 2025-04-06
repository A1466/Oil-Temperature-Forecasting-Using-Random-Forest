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
```

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

### ✅ Validation Set Results

```
Validation MAE: 0.05  
Validation RMSE: 0.09  
R² Score: 1.00
```

### ✅ Test Set Results

```
Test MAE: 0.06  
Test RMSE: 0.09  
R² Score: 0.99
```

---

## 📈 Visualizations

### 1. Validation: Actual vs Predicted

```python
plt.plot(val_dates, y_val.values, label='Actual OT')
plt.plot(val_dates, y_pred, label='Predicted OT')
```

### 2. Test Data: Actual vs Predicted

```python
plt.figure(figsize=(15, 5))
plt.plot(test_df['date'], y_test.values, label='Actual OT', color='blue')
plt.plot(test_df['date'], y_test_pred, label='Predicted OT', color='red', alpha=0.8)
plt.title("Test Data: Actual vs Predicted OT")
plt.xlabel("Datetime")
plt.ylabel("Oil Temperature")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 3. 24-Hour Forecast Plot

```python
plt.plot(forecast_df['date'], forecast_df['OT_Predicted'], marker='o', label='24-Hour Forecast')
```

### 4. Historical OT Plot (with 3-month interval)

```python
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
```

---

## 🚀 How to Run

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/oil-temperature-forecasting.git
   ```

2. **Place the following files in the working directory**:
   - `train.csv`
   - `test.csv`

3. **Run the Jupyter notebook or Python script**:
   - Open the `.ipynb` file in Jupyter Notebook or VS Code.
   - Execute all cells step-by-step, or run the Python script file directly.

4. **Visualize the results**:
   - Graphs will be generated for:
     - Validation predictions vs actual
     - Test predictions vs actual
     - 24-hour OT forecast

5. **Evaluate and tweak the model**:
   - Modify hyperparameters, test other models, or tune features to improve accuracy.

---

## 📷 Output Graphs

- 📈 Validation: Actual vs Predicted
- 📈 Test: Actual vs Predicted
- 📈 24-Hour Forecast
- 📈 Historical OT Trend (3-Month Gaps)

---


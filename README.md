# ASHRAE Great Energy Predictor III

Predicting building energy consumption using machine learning and deep learning models on the ASHRAE Kaggle dataset. The project compares traditional ML approaches (KNN, Decision Tree) against sequential neural networks (LSTM, GRU) for forecasting meter readings across 4 energy types.

##  Dataset

Data from the [ASHRAE Kaggle Competition](https://www.kaggle.com/c/ashrae-energy-prediction) covering ~1,000 buildings across 16 sites with hourly meter readings.

**Meter types:** Electricity · Chilled Water · Steam · Hot Water

##  Preprocessing & Feature Engineering

- Merged building metadata and weather data with meter readings
- Fixed unit conversion bug: Site 0 electricity readings were in kBTU → converted to kWh (×0.2931)
- Winsorized outliers and handled missing weather values
- Label encoded categorical features (`primary_use`, `season`)
- Added time-based features: `hour`, `day`, `weekday`, `month`, `season`
- Applied `log1p` transformation on the target (`meter_reading`) to handle skew
- Used `SelectKBest` for feature selection (top 4 features)

## Models

| Model | Details |
|-------|---------|
| **KNN** | `KNeighborsRegressor`, k=10, tuned over range 10–50 via pipeline |
| **Decision Tree** | `DecisionTreeRegressor`, max_depth=14, random_state=10 |
| **LSTM** | 2-layer LSTM (128 units each), Dropout (0.2), BatchNorm, Dense(1) — trained with RMSprop, 30 epochs |
| **GRU** | 2-layer GRU (128 units each), Dropout (0.2), BatchNorm, Dense(1) — trained with RMSprop, 30 epochs |

Both LSTM and GRU used `EarlyStopping` and were trained on reshaped 3D input `(samples, 1, features)`.


## Evaluation

- **Metric:** RMSLE (Root Mean Squared Log Error)
- Secondary: R² Score for regression quality
- 80/20 train-validation split (`random_state=45`)


## Tech Stack

Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · LightGBM · Matplotlib · Seaborn



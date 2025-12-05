# Flight Price Prediction — Machine Learning Project

## Overview

This project develops a complete end-to-end machine learning system to predict flight ticket prices using historical flight data. The goal is to build a production-grade model capable of estimating prices based on features such as airline, route, stops, flight duration, departure time, and arrival time.

The dataset contains more than **10,600 flight booking records**, sourced from multiple domestic airline routes.

---

## Objectives

| Task                                      | Status    |
| ----------------------------------------- | --------- |
| Exploratory Data Analysis (EDA)           | Completed |
| Predictive Model Training                 | Completed |
| Model Comparison & Performance Evaluation | Completed |
| Project Challenges & Technical Report     | Completed |

---

## Dataset Information

**Total samples:** 10,683
**Total raw features:** 11
**Target variable:** `Price`

| Column Name     | Description                    |
| --------------- | ------------------------------ |
| Airline         | Airline carrier                |
| Date_of_Journey | Journey date                   |
| Source          | Origin city                    |
| Destination     | Destination city               |
| Route           | Flight route                   |
| Dep_Time        | Departure time                 |
| Arrival_Time    | Arrival time                   |
| Duration        | Total travel time              |
| Total_Stops     | Number of stops                |
| Additional_Info | Miscellaneous information      |
| Price           | Ticket price (target variable) |

---

## Feature Engineering

More than **40 engineered features** were generated from the raw columns, including:

* Journey Day, Month, Year, Weekday, Weekend
* Departure / Arrival Hour & Minute
* Time-of-Day categories (Morning / Afternoon / Evening / Night)
* Duration in minutes
* Is_Direct_Flight indicator
* Total_Stops encoded numerically
* Route insights and categorical encodings

Encoding Strategy:

* One-Hot Encoding for low-cardinality categorical features
* Label Encoding for high-cardinality features

---

## Machine Learning Workflow

Multiple regression algorithms were implemented and evaluated:

| Model                   | Status  |
| ----------------------- | ------- |
| Linear Regression       | Trained |
| Decision Tree Regressor | Trained |
| Random Forest Regressor | Trained |
| XGBoost Regressor       | Trained |
| LightGBM Regressor      | Trained |
| CatBoost Regressor      | Trained |

Each model was evaluated using:

* R² Score
* RMSE (Root Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)
* 5-Fold Cross-Validation

---

## Final Model Performance

After hyperparameter tuning, **XGBoost** achieved the highest performance.

| Metric   | Tuned XGBoost |
| -------- | ------------- |
| R² Score | 0.8772        |
| RMSE     | ₹1,525        |
| MAPE     | 12.09%        |
| CV-R²    | 0.8510        |

More than **55.6%** of predictions fall within **10% of the actual price**, and **32.1%** within **5%**.

---

## Saved Files for Deployment

The following artifacts are exported for production use:

| File                                   | Description                          |
| -------------------------------------- | ------------------------------------ |
| `flight_price_model_xgboost_tuned.pkl` | Final trained machine learning model |
| `feature_names.pkl`                    | Feature list for preprocessing       |
| `model_metadata.pkl`                   | Training metadata and configuration  |

---

## How To Use

1. Clone the repository
2. Load `feature_names.pkl` to generate transformed input data
3. Load `flight_price_model_xgboost_tuned.pkl` for prediction inference
4. Ensure preprocessing logic matches the feature engineering steps from the notebook

---

## Repository Structure

```
/
├── Flight_Price_Prediction_Complete_Capstone.ipynb
├── flight_price_model_xgboost_tuned.pkl
├── feature_names.pkl
├── model_metadata.pkl
├── README.md
└── data/
      └── Flight_Fare.xlsx
```

---

## Business Value

* Airlines can use this system to optimize dynamic pricing
* Travel companies can provide fare predictions for customer planning
* Users can estimate expected ticket costs before booking

---

## Challenges & Solutions

| Challenge                             | Solution                                                   |
| ------------------------------------- | ---------------------------------------------------------- |
| High cardinality categorical features | Hybrid encoding approach                                   |
| Time parsing and duration extraction  | Custom transformation functions                            |
| Model selection uncertainty           | Comparative evaluation of six ML models                    |
| Initial low accuracy (R² = 0.52)      | Hyperparameter tuning increased performance to R² = 0.8772 |

---

## Future Enhancements

* Deployment as a REST API using FastAPI or Flask
* Integration with live airline price scraper
* LSTM/Transformer-based time-series forecasting
* Web dashboard for prediction visualization

---

## Authors

Muhammed Sayees

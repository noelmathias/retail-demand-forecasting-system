import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import  mean_absolute_error, r2_score, mean_absolute_percentage_error
from preprocess import load_data, create_features

DATA_PATH = "data/raw/retail_sales.csv"

def train_model():
    df = load_data(DATA_PATH)
    df = create_features(df)

    train = df[df["date"] <= '2016-12-31']
    test = df[df["date"] > '2016-12-31']

    features = [
        'store', 'item', 'day_of_week', 'month', 'year', 'lag_1', 'lag_7', 'lag_30','rolling_mean_7','rolling_std_7']
    target = "sales"

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    xgb = XGBRegressor(
        n_estimators = 300,
        learning_rate = 0.05,
        max_depth= 6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X_train, y_train)

    import pandas as pd
    importance = pd.Series(xgb.feature_importances_, index=X_train.columns)
    importance.to_csv("models/feature_importance.csv")


    y_pred = xgb.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2%},")
    print(f"R2 Score: {r2:.4f}")

    joblib.dump(xgb, "models/demand_forecast_model.pkl")

    print("Model saved to models/demand_forecast_model.pkl")

if __name__ == "__main__":
    train_model()
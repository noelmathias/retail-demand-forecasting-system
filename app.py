import streamlit as st
import pandas as pd
import joblib

importance = pd.read_csv("models/feature_importance.csv", index_col=0)

xgb = joblib.load("models/demand_forecast_model.pkl")
st.title("Retail Sales Demand Forecasting")
st.write(
    """
    This app predicts the demand for a specific item in a specific store for the next day based on historical sales data.
    This system uses lag features, rolling statistics to forecast future demand.
    """
)

df = pd.read_csv("data/raw/retail_sales.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['store', 'item', 'date'])

st.sidebar.header("Select Forecast Parameters")
store_id = st.sidebar.selectbox("Store", sorted(df['store'].unique()))
item_id = st.sidebar.selectbox("Item", sorted(df['item'].unique()))

subset = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()

st.subheader("Historical Sales ")
st.line_chart(subset.set_index('date')['sales'])
st.write(f"Selected Store: {store_id}")
st.write(f"Selected Item: {item_id}")

def forecast_next_7_days(model, subset, store_id, item_id):
    # Implementation for forecasting next 7 days

    history = subset.copy()
    future_predictions = []

    for i in range(7):
        last_date = history['date'].max()
        next_date = last_date + pd.Timedelta(days=1)

        day_of_week = next_date.dayofweek
        month = next_date.month
        year = next_date.year

        lag_1 = history['sales'].iloc[-1]
        lag_7 = history['sales'].iloc[-7] 
        lag_30 = history['sales'].iloc[-30]

        rolling_mean_7 = history['sales'].iloc[-7:].mean()
        rolling_std_7 = history['sales'].iloc[-7:].std()

        X = pd.DataFrame([{
            'store':store_id,
            'item': item_id,
            'day_of_week': day_of_week,
            'month': month,
            'year': year,
            'lag_1': lag_1,
            'lag_7': lag_7,
            'lag_30': lag_30,
            'rolling_mean_7': rolling_mean_7,
            'rolling_std_7': rolling_std_7
        }])
        prediction = xgb.predict(X)[0]

        future_predictions.append({
            'date': next_date,
            'predicted_sales': round(prediction)
        })
        history = pd.concat([history, pd.DataFrame({
            'date': [next_date],
            'sales': [prediction]
        })])

    return pd.DataFrame(future_predictions)

if st.button("Predict Tommorrow's Demand"):

    last_date = subset['date'].max()
    next_date = last_date + pd.Timedelta(days=1)

    day_of_week = next_date.dayofweek
    month = next_date.month
    year = next_date.year

    lag_1 = subset['sales'].iloc[-1]
    lag_7 = subset['sales'].iloc[-7] 
    lag_30 = subset['sales'].iloc[-30]

    rolling_mean_7 = subset['sales'].iloc[-7:].mean()
    rolling_std_7 = subset['sales'].iloc[-7:].std()

    X = pd.DataFrame([{
        'store':store_id,
        'item': item_id,
        'day_of_week': day_of_week,
        'month': month,
        'year': year,
        'lag_1': lag_1,
        'lag_7': lag_7,
        'lag_30': lag_30,
        'rolling_mean_7': rolling_mean_7,
        'rolling_std_7': rolling_std_7
    }])
    
    prediction = xgb.predict(X)[0]

    st.metric(
        label="Predicted Sales for Tomorrow",
        value=f"{prediction:.2f} units"
    )

    st.info(f"Forecast Date: {next_date.date()}")


forecast_df = forecast_next_7_days(xgb, subset, store_id, item_id)
st.subheader("7-Day Sales Forecast")
st.line_chart(forecast_df.set_index('date')['predicted_sales'])

st.markdown("-------")
st.subheader("Model Insights")
st.write("Feature importance used by the forecasting model:")
st.bar_chart(importance)

st.markdown("-------")
st.caption(
    "Retail Demand Forecasting Dashboard  |  Built with XGBoost and Streamlit"
)


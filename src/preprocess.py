import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store', 'item', 'date'])
    return df

def create_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    #lag features
    df['lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
    df['lag_7'] = df.groupby(['store', 'item'])['sales'].shift(7)
    df['lag_30'] = df.groupby(['store', 'item'])['sales'].shift(30)

    #rolling features
    df['rolling_mean_7'] = (
        df.groupby(['store', 'item'])['sales'].shift(1).rolling(7).mean()
    )

    df['rolling_std_7'] = (
        df.groupby(['store', 'item'])['sales'].shift(1).rolling(7).std()
    )

    df = df.dropna()
    return df

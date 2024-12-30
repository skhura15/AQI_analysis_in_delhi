#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima


# 1. Load and Preprocess Data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert 'date' column to datetime and extract year/month
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # Clean pollutant columns
    pollutant_columns = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    for col in pollutant_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract year and month
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Drop rows with missing pollutant data
    df.dropna(subset=pollutant_columns, inplace=True)
    return df

# 2. Plot PM2.5 Over Time
def plot_pm25_over_time(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['pm25'], color='red', label='PM2.5 (Raw Data)')
    plt.title("PM2.5 Levels Over Time")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 Levels (µg/m³)")
    plt.legend()
    plt.grid()
    plt.show()

# 3. Decompose Time Series
def decompose_time_series(df):
    df.set_index('date', inplace=True)
    result = seasonal_decompose(df['pm25'], model='additive', period=12)
    result.plot()
    plt.suptitle('Time Series Decomposition: PM2.5 Levels', y=1.02)
    plt.show()
    df.reset_index(inplace=True)

# 4. Train AutoARIMA Model for Monthly Forecasting
def train_autoarima_forecast(df):
    df.set_index('date', inplace=True)
    pm25_monthly = df['pm25'].resample('M').mean()
    
    # Handle NaN values: Fill with interpolation or drop NaNs
    pm25_monthly = pm25_monthly.interpolate(method='linear')  # Linear interpolation
    pm25_monthly = pm25_monthly.dropna()  # Ensure no NaNs remain
    
    # AutoARIMA to determine the best model
    auto_model = auto_arima(pm25_monthly, seasonal=True, m=12, trace=True, error_action="ignore", suppress_warnings=True)
    print(auto_model.summary())
    
    # Forecast for 2025
    forecast_steps = 12  # 12 months in 2025
    forecast, conf_int = auto_model.predict(n_periods=forecast_steps, return_conf_int=True)
    forecast_index = pd.date_range(start=pm25_monthly.index[-1] + pd.offsets.MonthBegin(), periods=forecast_steps, freq='M')
    
    # Plot Forecast
    plt.figure(figsize=(10, 6))
    plt.plot(pm25_monthly.index, pm25_monthly, label='Historical PM2.5', color='blue')
    plt.plot(forecast_index, forecast, label='Forecasted PM2.5 (2025)', color='orange')
    plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='orange', alpha=0.2, label="Confidence Interval")
    plt.title("Monthly Forecasted PM2.5 Levels for 2025 (AutoARIMA)")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 Levels (µg/m³)")
    plt.legend()
    plt.grid()
    plt.show()
    df.reset_index(inplace=True)


# 5. Correlation Heatmap
def correlation_heatmap(df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

# 6. Monthly Average PM2.5 Levels by Year
def plot_monthly_pm25(df):
    monthly_avg = df.groupby(['year', 'month'])['pm25'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    for year in sorted(df['year'].unique()):
        subset = monthly_avg[monthly_avg['year'] == year]
        plt.plot(subset['month'], subset['pm25'], label=str(year))
    plt.title("Monthly Average PM2.5 Levels by Year")
    plt.xlabel("Month")
    plt.ylabel("Average PM2.5 Levels (µg/m³)")
    plt.legend(title="Year")
    plt.grid()
    plt.show()

# 7. 3D Scatter Plot
def interactive_3d_plot(df):
    monthly_avg = df.groupby(['year', 'month'])['pm25'].mean().reset_index()
    fig = px.scatter_3d(monthly_avg, x='month', y='year', z='pm25', color='pm25',
                        labels={'month': 'Month', 'year': 'Year', 'pm25': 'PM2.5 Levels'},
                        title="3D Visualization of PM2.5 Levels Across Months and Years")
    fig.show()

# Main Function
def main():
    file_path = 'iti-jahangirpuri, delhi-air-quality.csv'  
    df = load_and_preprocess(file_path)
    print("Data Loaded and Preprocessed Successfully!")
    
    # Visualizations
    plot_pm25_over_time(df)      # PM2.5 over time
    decompose_time_series(df)    # Time series decomposition
    correlation_heatmap(df)      # Correlation matrix
    plot_monthly_pm25(df)        # Monthly PM2.5 levels
    interactive_3d_plot(df)      # 3D Scatter plot
    
    # Forecast PM2.5 using SARIMA
    train_autoarima_forecast(df)

if __name__ == "__main__":
    main()


# In[ ]:





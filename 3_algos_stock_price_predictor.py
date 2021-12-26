# import libraries
from google.protobuf.descriptor import Error
import streamlit as st

from datetime import date
import datetime as dt
import pandas_datareader as web
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import norm

# prophet is the facebook's prediction algorithm.
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Machine Learning and Statistics Frameworks.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# title of the web app
st.title("Stock Forecasting & Option Price Model")

st.subheader("Black-Scholes Option Price model")
### Black Scholes model
# Current stock (or other underlying) price

S = (st.text_input("Current Stock Price", 30.5))

# Strike price
K = (st.text_input("Strike Price of the option", 60))

# risk free interest rate
r = (st.text_input("10 year risk free interest rate (1.4 for 1.4%)", 1.4))

# time to maturity
t = (st.text_input("time to maturity in days", 394))

# Standard Deviation σ same as Implied Volatility for the option
sigma = (st.text_input("Implied Volatility (40 for 40%, 55 for 55%)", 44))

option_type = st.selectbox("Select Option Type (Call/Put)", ['Call', "Put"])

# Black-Schloles formula/function
def black_scholes_model(r, S, K, t, sigma, option_type):
    """Calculate BS option price for a call/put"""
    # checking if value is a string or None
    try:
        S = float(S)
        K = float(K)
        r = float(r) / 100
        t = float(t) / 365
        sigma = float(sigma) /100
        d1 = (np.log(S/K) + (r + sigma**2/2)*t)/(sigma*np.sqrt(t))
        d2 = d1 - sigma*np.sqrt(t)
    except TypeError as typ:
        st.write("Please check that you did not input text or empty space.")
    try:
        if (option_type == "Call"):
            price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*t)*norm.cdf(d2, 0, 1)
        elif (option_type == "Put"):
            price = K*np.exp(-r*t)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        return st.write(f"Fair Option Price: {price}")
    except:
        st.write("Please confirm all option parameters above!!!")

st.button('Generate Fair Price of Stock Option', on_click=black_scholes_model(r, S, K, t, sigma, option_type))


# Comparison of Financial Models to predict stock prices
# Statistical Models
# 1. ARIMA = Auto-Regressive Integrated Moving Average
# 2. stochastic process-geometric Brownian motion
# 3. Simple Linear Regression
# 4. Decison Tree

# Machine Learning Models
# 5. Artificial Neural Network = Sequential model
# 6. Prophet Facebook time-series predicting algorithm

# Measurement formulas used to calculate the error between predicted and actual price
# MAPE = Mean Absolute Percentage Error
# RMSE = Root Mean Squared Error
# AAE = Average Absolute Error

# The model uses the features (High, Low, Open, Volume, Adj Close) to predict the Close Price of the stock

st.subheader("Stock Price Forecasting Model")

# Downloading the data
selected_ticker = st.text_input("Ticker of the stock", "VIAC")

# 10 Year time period
start_final = dt.date(2011,1,1)
end_final = dt.datetime.now().date()

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(selected_ticker):
    data = web.DataReader(selected_ticker, 'yahoo', start_final, end_final)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
price_data_df = load_data(selected_ticker)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(price_data_df.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=price_data_df['Date'], y=price_data_df['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=price_data_df['Date'], y=price_data_df['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = price_data_df[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader("Facebook's Prophet Stock Price Forecasting Model In Use")
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast, xlabel='Date', ylabel='Value')
st.plotly_chart(fig1)












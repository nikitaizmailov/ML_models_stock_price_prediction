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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
st.title("Predict and Forecast Stock Prices in to the Future (via ML Models) & Black-Scholes Option Pricing Model")
st.subheader("Please Scroll Down To See All The Forecasting Models")
with st.container():
    st.subheader("Black-Scholes Option Pricing Model")
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
    S = float(S)
    K = float(K)
    r = float(r) / 100
    t = float(t) / 365
    sigma = float(sigma) /100
    d1 = (np.log(S/K) + (r + sigma**2/2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    if (option_type == "Call"):
        price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*t)*norm.cdf(d2, 0, 1)
    elif (option_type == "Put"):
        price = K*np.exp(-r*t)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    return st.write(f"Fair Option Price: {price}")

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

st.markdown("#")
st.subheader("Stock Price Forecasting Model")

# Downloading the data
selected_ticker = st.text_input("Ticker of the stock", "VIAC")

# 10 Year time period
start_final = dt.date(2011,1,1)
end_final = dt.datetime.now().date()

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365
st.write(f"Selected years of forecast into the future: {n_years}")

@st.cache
def load_data(selected_ticker):
    data = web.DataReader(selected_ticker, 'yahoo', start_final, end_final)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
price_data_df = load_data(selected_ticker)
data_load_state.text('Loading data... done!')

st.subheader('Downloaded Historical Stock Price Data from Yahoo Finance:')
st.write(price_data_df.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=price_data_df['Date'], y=price_data_df['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=price_data_df['Date'], y=price_data_df['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Forecast stock prices with Facebook's Prophet model.
df_train = price_data_df[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Facebook "Prophet" ML model: Predicting Stock Prices')
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast, xlabel='Date', ylabel='Value')
st.plotly_chart(fig1, use_container_width=True)


# Forecast stock price with neural network model.
# Artificial Neural Network
# Y-variable/Target variable is "Close price"
target_variable = price_data_df[['Close']]
# hyperparameters used to predict the close price
params = ["High", "Low", "Open", "Volume", "Adj Close"]

# Preprocess data pipeline
all_columns = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]

st.subheader("Artificial Neural Network Model's predicted and actual stock prices chart")

price_data_df2 = price_data_df.reset_index().drop("Date", axis=1)

full_pipeline_final = ColumnTransformer(transformers=[
    ('imputer', SimpleImputer(strategy='median'), all_columns),
    ('min_max_scl', MinMaxScaler(feature_range=(0,1)), all_columns),
],remainder='drop')

preped_dataframe = full_pipeline_final.fit_transform(price_data_df2)

preped_dataframe_df = pd.DataFrame(preped_dataframe)

preped_dataframe_final = preped_dataframe_df.iloc[:, 6:]
preped_dataframe_final.columns = all_columns
preped_dataframe_final.drop("Close", axis=1, inplace=True)

# Splitting the series into a training and test sets
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(preped_dataframe_final):
        X_train, X_test = preped_dataframe_final[:len(train_index)], preped_dataframe_final[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = target_variable[:len(train_index)].values.ravel(), target_variable[len(train_index): (len(train_index)+len(test_index))].values.ravel()

# LSTM requires any data being inputted to be 3 dimensional.
trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the model
# Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

# Model training
history = lstm.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1, shuffle=False)

# LSTM Prediction
y_pred = lstm.predict(X_test)

def convert_to_dfs(y_test_ser, y_pred_ser):
    # Just covnerting these two numpy arrays into dataframes
    y_test_df = pd.DataFrame(y_test_ser, columns=["Close Price"], index=price_data_df[len(train_index): (len(train_index)+len(test_index))]['Date'])
    y_pred_df = pd.DataFrame(y_pred_ser, columns=["Close Price"], index=price_data_df[len(train_index): (len(train_index)+len(test_index))]['Date'])

    return y_test_df, y_pred_df

y_test_plot, y_pred_plot = convert_to_dfs(y_test, y_pred)

# Dataframe with predicted, actual and Errors Metrics: RMSE and MAPE
def display_error_df(y_test_df_final, y_pred_df_final):
    error_combined_df= pd.merge(y_test_df_final, y_pred_df_final, left_index=True, right_index=True)
    error_combined_df.columns = ['Close Price (Actual)', 'Close Price (Predicted)']

    error_combined_df['Absolute Error'] = error_combined_df['Close Price (Predicted)'] - error_combined_df['Close Price (Actual)']

    error_mse = mean_squared_error(y_test, y_pred)
    error_rmse = np.sqrt(error_mse)
    error_combined_df['Root_Mean_Squared_Error'] = error_rmse

    error_mape = mean_absolute_percentage_error(y_test, y_pred)
    error_mape = error_mape * 100
    error_combined_df['Mean_Absolute_Percentage_Error'] = error_mape
    error_combined_df = error_combined_df.reset_index()
    return error_combined_df

error_df = display_error_df(y_test_plot, y_pred_plot)

st.subheader("Forecast data")
st.write(error_df.tail())

#Predicted vs True Adj Close Value – LSTM
def plot_fig(y_test_plot_ch, y_pred_plot_ch, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test_plot_ch.index, y=y_test_plot_ch['Close Price'], name="Actual Close Price"))
    fig.add_trace(go.Scatter(x=y_pred_plot_ch.index, y=y_pred_plot_ch['Close Price'], name="Predicted Close Price"))
    fig.layout.update(title_text=title, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_fig(y_test_plot, y_pred_plot, "Neural Network Model's predicted and actual prices")

# Linear Regression Model, Trained on train set and Tested on the test set.
st.subheader("Linear Regression Model's predicted and actual stock prices")
lin_reg = LinearRegression()

# reshaping the array from 3d to 2d.
X_train_reshaped = X_train.reshape(X_train.shape[0], (X_train.shape[1] * X_train.shape[2]))

# training the Linear Regression Model.
lin_reg.fit(X_train_reshaped, y_train)
X_test_reshaped = X_test.reshape(X_test.shape[0], (X_test.shape[1] * X_test.shape[2]))

# predicting via linear regression model, the stock prices
y_predicted_prices = lin_reg.predict(X_test_reshaped)

# getting data to plot.
y_test_plot2, y_pred_plot2 = convert_to_dfs(y_test, y_predicted_prices)

st.subheader("Forecast data")
# displaying the dataframe.
error_df = display_error_df(y_test_plot2, y_pred_plot2)
st.write(error_df.tail())

# plotting chart.
plot_fig(y_test_plot2, y_pred_plot2, "Linear Regression Model's predicted and actual prices")

# Forest Regression Tree
st.subheader("Decision Tree Model's predicted and actual stock prices.")
# Forest Regression Tree Model
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train_reshaped, y_train)

y_predicted_decision_tree = forest_reg.predict(X_test_reshaped)

# getting data to plot.
y_test_plot3, y_pred_plot3 = convert_to_dfs(y_test, y_predicted_decision_tree)

st.subheader("Forecast data")
# displaying the dataframe.
error_df = display_error_df(y_test_plot3, y_pred_plot3)
st.write(error_df.tail())

# plotting chart.
plot_fig(y_test_plot3, y_pred_plot3,  "Decison Tree Model's predicted and actual prices")






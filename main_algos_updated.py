# import libraries
from google.protobuf.descriptor import Error
import streamlit as st

from datetime import date
import datetime as dt
import pandas_datareader as web

import numpy as np
import pandas as pd

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
st.title("Stock Price Prediction Using Machine Learning Models")

st.subheader("Each model is unique and displays the predicted stock prices against actual stock prices, including into the future.")

st.write("Each model uses the last 50 days of the Close Price of the stock to predict the next day Close Price of the stock")
# Downloading the data
selected_ticker = st.text_input("Ticker of the stock", "AAPL")

# 10 Year time period
start_final = dt.date(2011,1,1)
end_final = dt.datetime.now().date()

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365
st.write(f"Selected years of forecast into the future: {n_years}")

def load_data(selected_ticker):
    data = web.DataReader(selected_ticker, 'yahoo', start_final, end_final)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
price_data_df = load_data(selected_ticker)
data_load_state.text('Loading data... done!')

st.subheader('Historical Stock Price Data sourced from Yahoo Finance:')
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
st.write('Generated Forecast data')
st.write(forecast.tail())

if n_years > 1:
    st.write(f'Forecast plot for {n_years} years')
else:
    st.write(f'Forecast plot for {n_years} year')
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
preped_dataframe_final.drop(params, axis=1, inplace=True)

preped_dataframe_final_used = np.array(preped_dataframe_final['Close'].tolist())

forecasted_days = 50

X_all_temp = []
y_all_temp = []

for x in range(forecasted_days, len(preped_dataframe_final_used)):
    # slicing first 50 elements and returning from each element only the 0 indexed position value. This way converting it from 2d array to 1d array.
    X_all_temp.append(preped_dataframe_final_used[x-forecasted_days:x])
    y_all_temp.append(target_variable.values[x])

train_index = int((len(X_all_temp) / 10) * 9)

X_test = X_all_temp[train_index:].copy()
X_train = X_all_temp[:train_index].copy()
y_test = y_all_temp[train_index:].copy()
y_train = y_all_temp[:train_index].copy()


y_test, y_train = np.array(y_test), np.array(y_train)


# LSTM requires any data being inputted to be 3 dimensional.
trainX = np.array(X_train)
testX = np.array(X_test)

X_train = trainX.reshape(len(X_train), 1, 50)
X_test = testX.reshape(len(X_test), 1, 50)

# Build the model
# Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, 50), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

# Model training
history = lstm.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1, shuffle=False)

# LSTM Prediction
y_pred = lstm.predict(X_test)

def convert_to_dfs(y_test_ser, y_pred_ser):
    # Just covnerting these two numpy arrays into dataframes
    y_test_df = pd.DataFrame(y_test_ser, columns=["Close Price"], index=price_data_df[train_index + forecasted_days: ]['Date'])
    y_pred_df = pd.DataFrame(y_pred_ser, columns=["Close Price"], index=price_data_df[train_index + forecasted_days: ]['Date'])

    return y_test_df, y_pred_df
y_test_plot, y_pred_plot = convert_to_dfs(y_test, y_pred)

# Dataframe with predicted, actual and Errors Metrics: RMSE and MAPE
def display_error_df(y_test_df_final, y_pred_df_final):
    error_combined_df= pd.merge(y_test_df_final, y_pred_df_final, left_index=True, right_index=True)
    error_combined_df.columns = ['Close Price', 'Predicted']

    error_combined_df['Absolute Error'] = error_combined_df['Predicted'] - error_combined_df['Close Price']

    error_mse = mean_squared_error(y_test_df_final, y_pred_df_final)
    error_rmse = np.sqrt(error_mse)
    error_combined_df['Root_Mean_Squared_Error'] = error_rmse

    error_mape = mean_absolute_percentage_error(y_test_df_final, y_pred_df_final)
    error_mape = error_mape * 100
    error_combined_df['Mean_Absolute_Percentage_Error'] = error_mape
    error_combined_df = error_combined_df.reset_index()
    return error_combined_df

error_df = display_error_df(y_test_plot, y_pred_plot)

st.write("Test Dataset: Actual and Predicted data")
st.write(error_df.tail())


#### These two functions below are used for the forecasting the stock prices into the future beyond the current date.
def generate_forecasted_data_future(model_used):
    # inputted data is last 50 days
    # scaled close price data
    final_input = preped_dataframe_final['Close'].values
    final_input = list(final_input[-50:])

    # to scale back the predicted value to be fed back into the model
    scaler = MinMaxScaler(feature_range=(0,1))
    # the slicing is done such way to keep the 2d format of (-1,1)
    scaler.fit(target_variable.values)

    x_inputs = []

    # variable to store the predicted values
    lst_output = []

    temp_input = []

    # using last 50 days to predict the next value
    forecasted_days = 50

    predict_days = period

    for x in range(forecasted_days, forecasted_days + predict_days):
        x_inputs.append(final_input[x-forecasted_days:x])
        x_input_temp = x_inputs[-1]
        x_input = np.array(x_inputs[-1])
        if not isinstance(model_used, Sequential):
            x_input = x_input.reshape(1, 50)
        else:
            x_input = x_input.reshape(1, 1, 50)
        temp_input.append(x_input_temp)
        y_future_val = model_used.predict(x_input)
        lst_output.append(float(y_future_val))
        y_future_val = y_future_val.reshape(-1,1)
        scaled_val = scaler.transform(y_future_val)
        final_input.append(float(scaled_val))
    
    return lst_output
        
def connecting_series(arr_forecasted, arr_predicted):
    final_array = list(arr_predicted.ravel())
    final_array.extend(arr_forecasted)
    
    return final_array

# Predicted vs True Adj Close Value – LSTM
def plot_fig(y_test_plot_ch, y_pred_plot_ch, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test_plot_ch.index, y=y_test_plot_ch['Close Price'], name="Actual Close Price"))
    fig.add_trace(go.Scatter(x=y_pred_plot_ch.index, y=y_pred_plot_ch['Close Price'], name="Predicted Close Price"))
    fig.layout.update(title_text=title, xaxis_rangeslider_visible=True, xaxis_title="Time Span in Days", yaxis_title=f"${selected_ticker} Share Price in USD")
    st.plotly_chart(fig)

# Dates and Variables below are used to forecast into the future stock prices.
# ANN model
t_i = generate_forecasted_data_future(lstm)

predicted_arr = connecting_series(t_i,y_pred)

date_temp = price_data_df[train_index + forecasted_days: ]['Date'].tolist()
start_temp = date_temp[0].date()
number_of_days = int(len(predicted_arr))
new_indices = pd.bdate_range(start=start_temp, periods=number_of_days)

def convert_to_dfs_for_prediction(y_test_ser, y_pred_ser):
    # Just covnerting these two numpy arrays into dataframes
    y_test_df = pd.DataFrame(y_test_ser, columns=["Close Price"])
    y_pred_df = pd.DataFrame(y_pred_ser, columns=["Close Price"])



    return y_test_df, y_pred_df

y_test_plot, y_pred_plot = convert_to_dfs_for_prediction(y_test, predicted_arr)

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

st.write("Test Dataset: Actual and Predicted data")
# displaying the dataframe.
error_df = display_error_df(y_test_plot2, y_pred_plot2)
st.write(error_df.tail())

# Dates and Variables below are used to forecast into the future stock prices.
# Linear Reg model
y_future_prices = generate_forecasted_data_future(lin_reg)

predicted_arr_2 = connecting_series(y_future_prices,y_predicted_prices)

# getting data to plot.
y_test_plot5, y_pred_plot5 = convert_to_dfs_for_prediction(y_test, predicted_arr_2)

# plotting chart.
plot_fig(y_test_plot5, y_pred_plot5, "Linear Regression Model's predicted and actual prices")







# Forest Regression Tree
st.subheader("Decision Tree Model's predicted and actual stock prices.")
# Forest Regression Tree Model
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train_reshaped, y_train)

y_predicted_decision_tree = forest_reg.predict(X_test_reshaped)

# getting data to plot.
y_test_plot3, y_pred_plot3 = convert_to_dfs(y_test, y_predicted_decision_tree)

st.write("Test Dataset: Actual and Predicted data")
# displaying the dataframe.
error_df = display_error_df(y_test_plot3, y_pred_plot3)
st.write(error_df.tail())

# Dates and Variables below are used to forecast into the future stock prices.
# Linear Reg model
y_future_prices2 = generate_forecasted_data_future(forest_reg)

predicted_arr_3 = connecting_series(y_future_prices2,y_predicted_prices)

# getting data to plot.
y_test_plot_updated, y_pred_plot_updated = convert_to_dfs_for_prediction(y_test, predicted_arr_3)

# plotting chart.
plot_fig(y_test_plot_updated, y_pred_plot_updated,  "Decison Tree Model's predicted and actual prices")



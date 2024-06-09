import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.express as px


st.write("# Stock Price Prediction | LSTM")

# Step 1: Allow the user to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Step 2: Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Make all column headers lowercase
    df.columns = df.columns.str.lower()
    
    # Display the dataframe
    st.write("### DataFrame")
    st.write(df)
    
    # Display df.describe()
    st.write("### Summary Statistics")
    st.write(df.describe())
    
    # Step 3: Plot a histogram of the 'close' prices
    st.write("### Histogram")
    # fig, ax = plt.subplots()
    # ax.hist(df['close'], bins=20)
    # st.pyplot(fig)
    # Create a histogram using Plotly
    fig = px.histogram(df, x='close', nbins=20, title='Histogram with Hover Information',
                    labels={'close': 'Close Price', 'count': 'Frequency'})

    # Add hover information
    fig.update_traces(hovertemplate='Bin Range: %{x}<br>Frequency: %{y}')

    # Display the histogram
    st.write("### Histogram with Hover Information (Using Plotly)")
    st.plotly_chart(fig)
    
    # Prepare the data for LSTM model
    df1 = df.reset_index()['close']
    
    # Plot the closing price
    st.write("### Closing Price Over Time")
    fig, ax = plt.subplots()
    ax.plot(df1)
    st.pyplot(fig)
    
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0,1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
    
    # Split the data
    training_size = int(len(df1) * 0.65)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]
    
    # Convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    # Reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    with st.spinner('model getting trained!!'):
    # Train the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12, batch_size=64, verbose=1)
    st.success('Done!')

    # Predict the prices
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    df1 = scaler.inverse_transform(df1)
    
    # Calculate RMSE
    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
    st.write(f"### Train RMSE: {train_rmse}")
    st.write(f"### Test RMSE: {test_rmse}")
    
    # Plotting
    look_back = 100
    train_predict_plot = np.empty_like(df1)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
    
    test_predict_plot = np.empty_like(df1)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    
    # Plot baseline and predictions
    st.write("### Predicted vs Actual Closing Prices")
    fig, ax = plt.subplots()
    ax.plot(df1, label="Actual Price")
    ax.plot(train_predict_plot, label="Train Prediction")
    ax.plot(test_predict_plot, label="Test Prediction")
    ax.legend()
    st.pyplot(fig)

    # Predicting the next 10 days
    st.write("### Predicting the Next 10 Days")

    # Get the last 100 days of data for prediction
    temp_input = test_data[-100:].flatten().tolist()

    lst_output = []
    n_steps = 100
    i = 0
    while(i < 10):  # Predicting for the next 10 days
        if(len(temp_input) > 100):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = np.array(temp_input).reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1

    # Transform the predictions back to the original scale
    predicted_prices = scaler.inverse_transform(lst_output)

    # Prepare data for Plotly
    last_100_days = scaler.inverse_transform(test_data[-100:])
    days_new = np.arange(1, 101)
    days_pred = np.arange(101, 111)

    # Create interactive plot with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=days_new, y=last_100_days.flatten(),
                            mode='lines', name='Last 100 Days'))

    fig.add_trace(go.Scatter(x=days_pred, y=predicted_prices.flatten(),
                            mode='lines', name='Next 10 Days'))

    fig.update_layout(
        title="Stock Price Prediction",
        xaxis_title="Days",
        yaxis_title="Price",
        hovermode='x unified'
    )

    st.plotly_chart(fig)

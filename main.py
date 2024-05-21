import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
st.write("# Stock Price Prediction | LSTM")

# Choose a stock
selection = st.selectbox( "Choose Stock", ['RELIANCE','HDFCBANK','BAJAJFINSV'])

st.write("### "+selection)

df = pd.read_csv("dataset/"+selection+".csv")

# Display the dataframe
st.write("### DataFrame")
st.write(df)

# Display df.describe()
st.write("### Summary Statistics")
st.write(df.describe())

st.write("### Histogram")
fig, ax = plt.subplots()
ax.hist(df['Close'], bins=20)
st.pyplot(fig)



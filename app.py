import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from transformers import pipeline
import torch
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
import warnings

warnings.filterwarnings("ignore")

# NewsAPI Key
NEWSAPI_KEY = "3087034a13564f75bfc769c0046e729c"

# ✅ Summarizer - PyTorch only
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

summarizer = load_summarizer()

# UI
st.title("🛢️ Crude Oil Research Dashboard")
crude_choice = st.selectbox("Select Crude", ["CL=F", "BZ=F"])
date_range = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"])

# Get prices
with st.spinner("Fetching prices..."):
    df = yf.download(crude_choice, period=date_range)
    df.reset_index(inplace=True)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines+markers'))
fig.update_layout(title="Crude Prices", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Rolling
df['Rolling5'] = df['Close'].rolling(5).mean()
st.line_chart(df[['Close', 'Rolling5']].set_index(df['Date']))

# Research
if st.button("Why up or down?"):
    with st.spinner("Searching news..."):
        url = f"https://newsapi.org/v2/everything?q=crude oil&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
        res = requests.get(url)
        articles = res.json().get("articles", [])
        text = " ".join([a['description'] or "" for a in articles[:5]])
        summary = summarizer(text, max_length=100, min_length=50)[0]['summary_text']
        st.write(summary)
        for a in articles:
            st.write(f"[{a['title']}]({a['url']})")

# Forecast
st.subheader("⏳ LSTM Forecast")
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Close']].values)

look_back = 5
X, y = [], []
for i in range(look_back, len(scaled)):
    X.append(scaled[i-look_back:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)

model = Sequential()
model.add(LSTM(30, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

with st.spinner("Training model..."):
    model.fit(X, y, epochs=1, batch_size=1, verbose=0)

inputs = scaled[-look_back:]
future = []
for _ in range(10):
    pred = model.predict(inputs.reshape(1, look_back, 1), verbose=0)
    future.append(pred[0][0])
    inputs = np.vstack((inputs[1:], pred))

forecast = scaler.inverse_transform(np.array(future).reshape(-1, 1))
dates = pd.date_range(start=df['Date'].iloc[-1], periods=11, freq='D')[1:]
df_forecast = pd.DataFrame({'Date': dates, 'Forecast': forecast.flatten()})
st.line_chart(df_forecast.set_index('Date'))


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from transformers import pipeline
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests

# 🔒 Ignore protobuf + tf warnings
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------
# 🔑 Hardcoded NewsAPI Key
NEWSAPI_KEY = "3087034a13564f75bfc769c0046e729c"

# ------------------------------------------
# 1️⃣ Summarizer (PyTorch only)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

summarizer = load_summarizer()

# ------------------------------------------
# 2️⃣ User selects crude & range
st.title("🌍 Crude Oil Price Trend + Research + Forecast")
crude_options = ["CL=F", "BZ=F", "GC=F"]
crude_choice = st.selectbox("Select Crude:", crude_options)
date_range = st.selectbox(
    "Select Range:", ["1mo", "3mo", "6mo", "1y", "5y"]
)

# ------------------------------------------
# 3️⃣ Load prices
data = yf.download(crude_choice, period=date_range, interval="1d")
if data.empty:
    st.error("Could not load crude data.")
    st.stop()

data.reset_index(inplace=True)
st.subheader("📈 Price Trend")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data['Date'], y=data['Close'],
    mode='lines+markers',
    name=f"{crude_choice} Close Price"
))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x",
    xaxis_rangeslider_visible=True
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# 4️⃣ Detailed rolling average hover
data['Rolling5'] = data['Close'].rolling(window=5).mean()
st.line_chart(data[['Close', 'Rolling5']].set_index(data['Date']))

# ------------------------------------------
# 5️⃣ News Research
if st.button("🔍 Research: Why up or down?"):
    st.info("Fetching articles...")
    query = f"{crude_choice} crude oil news"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize=10&apiKey={NEWSAPI_KEY}"
    res = requests.get(url)
    if res.status_code == 200:
        articles = res.json().get("articles", [])
        text = " ".join([a['description'] or "" for a in articles])
        if text.strip():
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
            st.subheader("📑 Research Summary")
            st.write(summary[0]['summary_text'])
            st.subheader("🔗 Articles:")
            for a in articles:
                st.write(f"[{a['title']}]({a['url']})")
        else:
            st.warning("No descriptions found.")
    else:
        st.error(f"Error fetching news: {res.status_code}")

# ------------------------------------------
# 6️⃣ Forecast with LSTM
st.subheader("⏳ Forecast Next 14 Days")

# Prepare data
data_forecast = data[['Date', 'Close']].dropna()
data_forecast.index = data_forecast['Date']
data_forecast = data_forecast[['Close']]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_forecast)

look_back = 5
X_train, y_train = [], []

for i in range(look_back, len(scaled_data)-1):
    X_train.append(scaled_data[i-look_back:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=0)

inputs = scaled_data[-look_back:]
preds = []

for _ in range(14):
    input_reshaped = np.reshape(inputs, (1, look_back, 1))
    pred = model.predict(input_reshaped, verbose=0)
    preds.append(pred[0,0])
    inputs = np.append(inputs[1:], pred, axis=0)

forecast_prices = scaler.inverse_transform(np.array(preds).reshape(-1,1))
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=15, freq='D')[1:]

df_forecast = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_prices.flatten()})
st.line_chart(df_forecast.set_index('Date'))

st.success("✅ Done!")

# ------------------------------------------
st.caption("🔬 Built with PyTorch NLP + LSTM forecasting.")

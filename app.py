
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from transformers import pipeline
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="🛢️ Crude Oil LSTM Researcher", layout="wide")
st.title("🛢️ Crude Oil Price Trend Researcher - LSTM Advanced")

NEWSAPI_KEY = "3087034a13564f75bfc769c0046e729c"

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("⚙️ Settings")

crude = st.sidebar.selectbox(
    "Select Crude:",
    ["Brent", "WTI"], index=0
)
symbol = {"Brent": "BZ=F", "WTI": "CL=F"}[crude]

time_label = st.sidebar.radio(
    "Time Range:",
    ["1M", "3M", "6M", "1Y", "5Y"], index=0
)
time_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
period = time_map[time_label]

# ------------------------------
# FETCH PRICE DATA
# ------------------------------
@st.cache_data(ttl=3600)
def get_prices(symbol, period):
    df = yf.download(symbol, period=period, interval="1d")
    if len(df) < 5:
        df = yf.download(symbol, period=period, interval="1h")
    df.reset_index(inplace=True)
    df['Rolling_5D'] = df['Close'].rolling(5).mean()
    df.dropna(inplace=True)
    return df

df = get_prices(symbol, period)

if df.empty or len(df) < 10:
    st.error("⚠️ Not enough data to analyze. Try a larger time range.")
else:
    # ------------------------------
    # PLOT PRICES
    # ------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines+markers', name='Close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Rolling_5D'], mode='lines', name='5-Day Avg', line=dict(dash='dot')))
    fig.update_layout(
        title=f"{crude} Crude Oil Prices",
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# MARKET RESEARCH
# ------------------------------
st.subheader("🔍 Why did prices move?")

if st.button("Run Deep Research"):
    today = datetime.today().strftime('%Y-%m-%d')
    week_ago = (datetime.today() - timedelta(days=14)).strftime('%Y-%m-%d')
    url = f"https://newsapi.org/v2/everything?q={crude}+crude+oil&from={week_ago}&to={today}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&pageSize=100"

    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if not articles:
            st.warning("No relevant news found.")
        else:
            full_text = ". ".join([f"{a['title']}. {a.get('description','')}" for a in articles])
            full_text = full_text[:3500]  # Hugging Face limit

            summary = summarizer(full_text, max_length=250, min_length=120, do_sample=False)[0]['summary_text']
            st.success("### 📜 Detailed Market Insight:")
            st.write(summary)

            st.info("### 🔗 Top Sources:")
            for a in articles[:10]:
                st.markdown(f"- [{a['title']}]({a['url']})")
    else:
        st.error("Error fetching news. Check your API key or quota.")

# ------------------------------
# LSTM FORECAST
# ------------------------------
st.subheader("📈 Forecast Future Prices (LSTM)")

if st.button("Run LSTM Forecast"):
    if df.empty or len(df) < 50:
        st.warning("❌ Not enough data for LSTM. Try 6M or 1Y.")
    else:
        # Prepare Data
        data = df[['Date', 'Close']].copy()
        data.index = data['Date']
        data.drop(['Date'], axis=1, inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create training data
        train_data = scaled_data
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train
        with st.spinner('Training LSTM... (This may take a minute)'):
            model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

        # Forecast next 30 days
        forecast_days = st.slider("Forecast days ahead:", 7, 90, 30)
        inputs = scaled_data[-60:]
        predictions = []
        for _ in range(forecast_days):
            X_test = np.reshape(inputs, (1, inputs.shape[0], 1))
            pred_price = model.predict(X_test)
            predictions.append(pred_price[0,0])
            inputs = np.append(inputs, [[pred_price[0,0]]], axis=0)
            inputs = inputs[1:]

        forecast_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=forecast_days+1, freq='B')[1:]

        # Plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical'))
        fig2.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices.flatten(), mode='lines', name='LSTM Forecast'))
        fig2.update_layout(
            title=f"{crude} Crude Oil LSTM Forecast",
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)

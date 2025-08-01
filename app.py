# -----------------------------------------------------------
# CRUDE OIL TREND ANALYZER - ADVANCED VERSION - LSTM + DEEP NEWS INSIGHT
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="🛢️ Crude Oil Researcher", layout="wide")
st.title("🛢️ Crude Oil Price Researcher - Next-Gen Version")

NEWSAPI_KEY = "3087034a13564f75bfc769c0046e729c"

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

summarizer = load_summarizer()

# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.header("⚙️ Settings")

crude = st.sidebar.selectbox(
    "Select Crude:",
    ["Brent", "WTI"]
)
symbol_map = {"Brent": "BZ=F", "WTI": "CL=F"}
symbol = symbol_map[crude]

time_label = st.sidebar.radio(
    "Time Range:",
    ["1M", "3M", "6M", "1Y", "5Y"], index=1
)
time_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
period = time_map[time_label]

show_volume = st.sidebar.checkbox("Show Volume", value=True)

# -----------------------------------------------------------
# FETCH PRICE DATA
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def get_prices(symbol, period):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True)
    df.reset_index(inplace=True)
    df['Rolling_5D'] = df['Close'].rolling(window=5).mean()
    df.dropna(inplace=True)
    return df

df = get_prices(symbol, period)

if df.empty or len(df) < 5:
    st.error("⚠️ Not enough data. Try larger time range.")
    st.stop()

# -----------------------------------------------------------
# PLOT PRICES
# -----------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines+markers', name='Close Price'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Rolling_5D'], mode='lines', name='5D Rolling Avg', line=dict(dash='dot')))
if show_volume:
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', yaxis='y2'))

fig.update_layout(
    title=f"{crude} Crude Oil Price Trend",
    hovermode='x',
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
    ),
    yaxis=dict(title="Price (USD)"),
    yaxis2=dict(overlaying='y', side='right', showgrid=False, title="Volume")
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# DEEP RESEARCH ANALYSIS
# -----------------------------------------------------------
st.subheader("🔍 Market Intelligence: Why is price moving?")

if st.button("Run Advanced Research"):
    today = datetime.today().strftime('%Y-%m-%d')
    week_ago = (datetime.today() - timedelta(days=14)).strftime('%Y-%m-%d')
    url = f"https://newsapi.org/v2/everything?q={crude}+crude+oil+OR+OPEC+OR+geopolitics+OR+sanctions&from={week_ago}&to={today}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&pageSize=100"

    r = requests.get(url)
    if r.status_code == 200:
        articles = r.json().get('articles', [])
        if not articles:
            st.warning("No news found.")
        else:
            all_texts = []
            for a in articles:
                snippet = f"{a['title']}. {a.get('description','')}. {a.get('content','')}"
                all_texts.append(snippet)

            big_text = " ".join(all_texts)
            big_text = big_text[:4500]  # handle max token limit

            summary = summarizer(big_text, max_length=400, min_length=150, do_sample=False)[0]['summary_text']
            st.success("### 📜 Detailed Market Analysis")
            st.write(summary)

            st.info("### 🔗 Top Headlines:")
            for a in articles[:15]:
                st.markdown(f"- [{a['title']}]({a['url']})")
    else:
        st.error("NewsAPI error. Check key or quota.")

# -----------------------------------------------------------
# LSTM FORECAST
# -----------------------------------------------------------
st.subheader("📈 Forecast Future Prices (LSTM)")

if st.button("Run LSTM Forecast"):
    if len(df) < 60:
        st.warning("❌ Not enough data. Try longer period.")
    else:
        dataset = df[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(dataset)

        X_train, y_train = [], []
        for i in range(60, len(scaled)):
            X_train.append(scaled[i-60:i, 0])
            y_train.append(scaled[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        with st.spinner('⏳ Training LSTM...'):
            model.fit(X_train, y_train, epochs=7, batch_size=32, verbose=0)

        # Predict future
        inputs = scaled[-60:]
        forecast = []
        for _ in range(30):
            X_test = np.reshape(inputs, (1, inputs.shape[0], 1))
            pred = model.predict(X_test)
            forecast.append(pred[0,0])
            inputs = np.append(inputs, [[pred[0,0]]], axis=0)
            inputs = inputs[1:]

        predicted_prices = scaler.inverse_transform(np.array(forecast).reshape(-1,1))
        future_dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical'))
        fig2.add_trace(go.Scatter(x=future_dates, y=predicted_prices.flatten(), mode='lines', name='Forecast'))
        fig2.update_layout(
            title=f"{crude} LSTM Forecast",
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)

st.caption("⚙️ Built with 💪 Streamlit, Yahoo Finance, HuggingFace Transformers, TensorFlow, NewsAPI.")



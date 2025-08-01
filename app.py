import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
import requests
from transformers import pipeline
from datetime import datetime, timedelta

# --------------------------------------------
# 1️⃣ APP CONFIG & KEYS
# --------------------------------------------

st.set_page_config(page_title="Crude Oil Price Researcher", layout="wide")

st.title("🛢️ Advanced Real-Time Crude Oil Trend Analyzer & Researcher")
st.write("""
This tool tracks global crude oil prices, forecasts future trends,  
and explains **why** the trends are happening using real-time news scraping and NLP.
""")

# ✅ Hardcoded API KEY - Replace with your own NewsAPI key!
NEWSAPI_KEY = "3087034a13564f75bfc769c0046e729c"

# Initialize NLP summarizer (transformers)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = load_summarizer()

# --------------------------------------------
# 2️⃣ USER SELECTION PANEL
# --------------------------------------------

st.sidebar.header("⚙️ Settings")

# Select Crude Oil Type
crude = st.sidebar.selectbox(
    "Select Crude Oil Benchmark:",
    ["Brent", "WTI"],
    index=0
)
symbol_map = {
    "Brent": "BZ=F",
    "WTI": "CL=F"
}
symbol = symbol_map[crude]

# Timeframe buttons & manual date range
st.sidebar.subheader("🗓️ Time Range")

time_options = {
    "1 Week": "7d",
    "2 Weeks": "14d",
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}
time_label = st.sidebar.radio(
    "Quick Select:",
    list(time_options.keys()),
    index=4
)
period = time_options[time_label]

# --------------------------------------------
# 3️⃣ FETCH PRICE DATA
# --------------------------------------------

@st.cache_data(ttl=3600)
def get_price_data(symbol, period):
    df = yf.download(symbol, period=period, interval='1d')
    df.reset_index(inplace=True)
    df['Rolling_5D'] = df['Close'].rolling(window=5).mean()
    df.dropna(inplace=True)
    return df

st.info(f"Fetching {crude} price data for: **{time_label}** ...")
df = get_price_data(symbol, period)

# --------------------------------------------
# 4️⃣ MAIN INTERACTIVE PLOT
# --------------------------------------------

st.subheader(f"📊 {crude} Crude Oil Trend")

fig = go.Figure()

# Main line
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'],
    mode='lines+markers',
    name='Daily Close',
    line=dict(color='royalblue', width=2)
))

# 5-day rolling avg
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Rolling_5D'],
    mode='lines',
    name='5-Day Rolling Avg',
    line=dict(color='orange', dash='dot')
))

# Chart options: time buttons, slider, hover mode
fig.update_layout(
    hovermode='x unified',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=14, label="2W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=2, label="2Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis_title="Price (USD/barrel)",
    title=f"{crude} Crude Oil Closing Prices"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# 5️⃣ SMART RESEARCH & NLP
# --------------------------------------------

st.subheader("🔍 Intelligent Research: Why did the trend move?")

if st.button("Run Market Research"):
    st.info("⏳ Fetching latest news & analyzing... Please wait.")
    today = datetime.today().strftime('%Y-%m-%d')
    week_ago = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')

    search_terms = f"{crude} crude oil price OR OPEC OR supply OR conflict OR energy policy"

    url = f"https://newsapi.org/v2/everything?q={search_terms}&from={week_ago}&to={today}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en"

    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if not articles:
            st.warning("⚠️ No relevant news found for the past week.")
        else:
            st.success(f"🔗 Found {len(articles)} related news articles.")
            combined_text = ""
            for art in articles[:5]:
                combined_text += f"{art['title']}. {art.get('description','')} "

            # Run NLP summarizer
            st.info("🤖 Running advanced NLP summarization ...")
            summary = summarizer(
                combined_text,
                max_length=250,
                min_length=100,
                do_sample=False
            )[0]['summary_text']

            st.write("### 📌 Key Drivers & Market Reasons:")
            st.write(summary)

            st.write("### 🔗 Sources:")
            for art in articles[:5]:
                st.markdown(f"- [{art['title']}]({art['url']})")
    else:
        st.error(f"❌ News API request failed. Status: {response.status_code}")

# --------------------------------------------
# 6️⃣ FORECASTING BLOCK
# --------------------------------------------

st.subheader("📈 Advanced Forecasting: Predict Future Prices")

if st.button("Run Forecast"):
    st.info("⏳ Training forecasting model...")
    data = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(data)

    # Select horizon
    horizon_days = st.slider("Select forecast horizon (days):", 7, 365, 30)
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    fig2 = go.Figure()

    # Actual
    fig2.add_trace(go.Scatter(
        x=data['ds'],
        y=data['y'],
        mode='lines',
        name='Actual'
    ))

    # Forecast line
    fig2.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='green')
    ))

    # Confidence intervals
    fig2.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig2.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.2)',
        name='Confidence Interval'
    ))

    fig2.update_layout(
        title=f"{crude} Forecast for Next {horizon_days} Days",
        hovermode='x unified',
        yaxis_title="Price (USD/barrel)"
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.info("✅ Your advanced research dashboard is ready! Enjoy the insights. 👑")

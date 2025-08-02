import streamlit as st
import pandas as pd
import spgci as ci
import requests
from transformers import pipeline
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --------------------------------------------
# 1️⃣ App Config
# --------------------------------------------

st.set_page_config(
    page_title="🛢️ Crude Oil Trend Analyzer with Platts",
    layout="wide"
)

st.title("🛢️ Real-Time Crude Oil Price Analyzer & Researcher (Platts Connect)")
st.write("""
This advanced dashboard connects **directly** to Platts Connect via the `spgci` SDK,  
pulls **live market data**, and runs AI-powered research using NewsAPI + Insights.
""")

# --------------------------------------------
# 2️⃣ Platts Connect Setup (Hardcoded for testing only!)
# --------------------------------------------

# ✅ Correct positional args — no keywords!
ci.set_credentials("sharugh.a@spglobal.com", "T!mezone22")

mdd = ci.MarketData()
ni = ci.Insights()

)

mdd = ci.MarketData()
ni = ci.Insights()

# --------------------------------------------
# 3️⃣ NLP Summarizer
# --------------------------------------------

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# --------------------------------------------
# 4️⃣ Sidebar Selection Panel
# --------------------------------------------

st.sidebar.header("⚙️ Settings")

# Get all crude oil symbols via Platts Connect
crude_symbols_df = mdd.get_symbols(commodity="Crude oil")
crude_symbols = crude_symbols_df['symbol'].unique().tolist()

selected_symbol = st.sidebar.selectbox(
    "Select Crude Oil Symbol from Platts:",
    crude_symbols
)

# Time range (simple: last X days)
st.sidebar.subheader("🗓️ Time Range")
days_back = st.sidebar.slider("Number of past days to view:", 7, 90, 30)

# --------------------------------------------
# 5️⃣ Fetch Market Data
# --------------------------------------------

st.info(f"🔄 Fetching market data for `{selected_symbol}` from Platts...")

# Pull assessments for selected symbol & MDC
mdcs_df = mdd.get_mdcs()
symbol_mdc = mdcs_df.iloc[0]['mdc']  # Use first MDC for demo — refine as needed

assessments_df = mdd.get_assessments_by_symbol(selected_symbol)
assessments_df['timestamp'] = pd.to_datetime(assessments_df['timestamp'])
assessments_df = assessments_df.sort_values('timestamp')

# Filter by time window
today = datetime.utcnow()
past_date = today - timedelta(days=days_back)
assessments_df = assessments_df[
    assessments_df['timestamp'] >= past_date
]

# If empty, warn
if assessments_df.empty:
    st.warning(f"⚠️ No data found for the last {days_back} days.")
else:
    st.success(f"✅ Retrieved {len(assessments_df)} price points.")

    # --------------------------------------------
    # 6️⃣ Price Chart
    # --------------------------------------------

    st.subheader(f"📈 Price Trend for `{selected_symbol}`")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=assessments_df['timestamp'],
        y=assessments_df['price'],
        mode='lines+markers',
        name='Platts Price',
        line=dict(color='royalblue', width=2)
    ))

    fig.update_layout(
        hovermode='x unified',
        title=f"Platts Price Assessment - {selected_symbol}",
        yaxis_title="Price (USD/barrel)",
        xaxis_title="Date",
        xaxis=dict(rangeslider=dict(visible=True))
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# 7️⃣ Smart Research Block — NewsAPI + Insights
# --------------------------------------------

st.subheader("🔍 AI Research: What’s driving the market?")

if st.button("Run Research"):
    with st.spinner("⏳ Fetching NewsAPI + Platts Insights..."):

        # ----------- NewsAPI -----------
        today = datetime.today().strftime('%Y-%m-%d')
        week_ago = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')

        search_terms = f"{selected_symbol} crude oil price OR OPEC OR supply OR conflict OR energy policy"
        NEWSAPI_KEY = "3087034a13564f75bfc769c0046e729c"  # ✅ Your real NewsAPI key

        url = (
            f"https://newsapi.org/v2/everything?q={search_terms}"
            f"&from={week_ago}&to={today}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en"
        )
        response = requests.get(url)

        news_articles = []
        if response.status_code == 200:
            news_articles = response.json().get('articles', [])
        else:
            st.warning("⚠️ NewsAPI failed. Check your API key.")

        # ----------- Platts Insights -----------
        insights_df = ni.get_stories(
            q=selected_symbol,
            content_type=ni.ContentType.MarketCommentary
        )

        # Combine all text
        combined_text = ""

        if news_articles:
            for art in news_articles[:5]:
                combined_text += f"{art['title']}. {art.get('description', '')} "

        if not insights_df.empty:
            for idx, row in insights_df.head(5).iterrows():
                combined_text += f"{row['headline']}. {row.get('content','')} "

        if not combined_text:
            st.warning("❌ No articles or stories found.")
        else:
            # NLP Summarizer
            st.info("🤖 Running NLP summarization ...")
            summary = summarizer(
                combined_text,
                max_length=250,
                min_length=100,
                do_sample=False
            )[0]['summary_text']

            st.write("### 📌 Summary Insights:")
            st.write(summary)

            # Sources — NewsAPI
            if news_articles:
                st.write("### 🔗 News Sources:")
                for art in news_articles[:5]:
                    st.markdown(f"- [{art['title']}]({art['url']})")

            # Sources — Platts
            if not insights_df.empty:
                st.write("### 🔗 Platts Insights:")
                for idx, row in insights_df.head(5).iterrows():
                    st.markdown(f"- [{row['headline']}]({row['url']})")

st.info("✅ All done! Platts Connect is powering your crude oil analysis 🚀")

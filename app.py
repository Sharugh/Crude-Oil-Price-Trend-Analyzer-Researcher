# --------------------------------------------
# Crude Oil Researcher with SPGCI (No Appkey)
# --------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import spgci as ci
from transformers import pipeline
import os

# --------------------------------------------
# 1️⃣ APP CONFIG
# --------------------------------------------

st.set_page_config(page_title="SPGCI Crude Oil Researcher", layout="wide")

st.title("🛢️ Real-Time Crude Oil Market Data & Insights")
st.write("""
This tool uses **S&P Global Commodity Insights (SPGCI)** to fetch real-time crude oil market data  
and summarize the latest expert commentary and signals.
""")

# --------------------------------------------
# 2️⃣ AUTH: set_credentials without Appkey
# --------------------------------------------

username = os.getenv("SPGCI_USERNAME")
password = os.getenv("SPGCI_PASSWORD")

# ✅ Explicitly call set_credentials WITHOUT Appkey
ci.set_credentials(username=username, password=password)

# --------------------------------------------
# 3️⃣ SPGCI CLIENTS
# --------------------------------------------

mdd = ci.MarketData()
ni = ci.Insights()

# --------------------------------------------
# 4️⃣ USER SELECTION PANEL
# --------------------------------------------

st.sidebar.header("⚙️ Settings")

commodity = "Crude oil"

# Show all symbols for crude oil
symbols_df = mdd.get_symbols(commodity=commodity)
symbols = symbols_df['symbol'].unique().tolist()
symbol = st.sidebar.selectbox("Select Crude Oil Symbol:", symbols)

# Show Market Data Categories
mdcs_df = mdd.get_mdcs(subscribed_only=True)
mdcs = mdcs_df['mdc'].unique().tolist()
mdc = st.sidebar.selectbox("Select Market Data Category (MDC):", mdcs)

# --------------------------------------------
# 5️⃣ GET MARKET DATA
# --------------------------------------------

st.subheader(f"📊 {commodity} Market Assessments")

assessments_df = mdd.get_assessments_by_mdc_current(mdc=mdc)
if assessments_df.empty:
    st.warning(f"No market assessments found for MDC: {mdc}")
else:
    st.dataframe(assessments_df)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=assessments_df['symbol'],
        y=assessments_df['value'],
        name='Assessment Value',
        marker_color='royalblue'
    ))

    fig.update_layout(
        title=f"Current Assessments for {mdc}",
        xaxis_title="Symbol",
        yaxis_title="Price/Assessment Value",
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# 6️⃣ INSIGHTS & NLP SUMMARIZATION
# --------------------------------------------

st.subheader("🔍 Latest SPGCI Insights for Crude Oil")

# Load NLP summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

if st.button("Run Insights Research"):
    st.info("⏳ Fetching latest commentary, notes, and heards ...")

    # 1️⃣ Market Commentary
    commentary_df = ni.get_stories(q=commodity, content_type=ni.ContentType.MarketCommentary)

    # 2️⃣ Subscriber Notes
    notes_df = ni.get_subscriber_notes(q=commodity)

    # 3️⃣ Heards
    heards_df = ni.get_heards(q=commodity, content_type=ni.HeardsContentType.Heard,
                              geography=['Global'], strip_html=True)

    combined_text = ""

    if not commentary_df.empty:
        st.success(f"✅ Market Commentary: {len(commentary_df)} stories found.")
        combined_text += " ".join(commentary_df['title'].astype(str).tolist())

    if not notes_df.empty:
        st.success(f"✅ Subscriber Notes: {len(notes_df)} notes found.")
        combined_text += " ".join(notes_df['title'].astype(str).tolist())

    if not heards_df.empty:
        st.success(f"✅ Heards: {len(heards_df)} heard reports found.")
        combined_text += " ".join(heards_df['title'].astype(str).tolist())

    if combined_text:
        st.info("🤖 Running summarization...")
        summary = summarizer(
            combined_text,
            max_length=250,
            min_length=100,
            do_sample=False
        )[0]['summary_text']

        st.write("### 📌 Market Insight Summary:")
        st.write(summary)

        st.write("### 🔗 Sources:")
        if not commentary_df.empty:
            for _, row in commentary_df.iterrows():
                st.markdown(f"- **Market Commentary:** {row['title']}")
        if not notes_df.empty:
            for _, row in notes_df.iterrows():
                st.markdown(f"- **Subscriber Note:** {row['title']}")
        if not heards_df.empty:
            for _, row in heards_df.iterrows():
                st.markdown(f"- **Heard:** {row['title']}")
    else:
        st.warning("⚠️ No relevant crude oil insights found at the moment.")

st.info("✅ Ready! Data powered by SPGCI.")



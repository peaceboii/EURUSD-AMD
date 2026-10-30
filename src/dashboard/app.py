import time
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="1H Market Phase Monitor", page_icon="📊", layout="wide"
)

API_URL = "http://127.0.0.1:8000"

st.title("📊 1-Hour Institutional Phase Monitor (EUR/USD)")
st.caption(
    "Live IBKR 1H Bars ➔ feature_eng.py ➔ XGBoost Model ➔ Phase Classification"
)

try:
    resp = requests.get(f"{API_URL}/state", timeout=2)
    data = resp.json()
except Exception:
    data = None
    st.error("⚠️ Cannot connect to backend API at http://127.0.0.1:8000.")

if data:
    # 1. Top Bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IBKR Socket", "CONNECTED" if data["connected"] else "OFFLINE")
    c2.metric("EUR/USD Spot", f"{data['spot_price']:.5f}")
    c3.metric("Last 1H Bar", data["last_bar_time"])
    c4.metric("Model Prediction", f"{data['predicted_target']:.4f}")

    st.markdown("---")

    # 2. Institutional Phase Banner
    phase = data["market_phase"]
    if "ACCUMULATION" in phase:
        st.info(f"### 🔵 Phase: {phase}")
        st.caption(
            "Market is coiling. Narrow candle range and contracting volatility."
        )
    elif "MANIPULATION" in phase:
        st.warning(f"### 🟠 Phase: {phase}")
        st.caption(
            "High wick ratio rejection. Smart money running liquidity outside key levels."
        )
    elif "DISTRIBUTION" in phase:
        st.success(f"### 🟢 Phase: {phase}")
        st.caption(
            "Institutional expansion. Large body displacement with elevated volume/volatility."
        )
    else:
        st.markdown(f"### ⚪ Phase: {phase}")

    st.markdown("---")

    # 3. Features Breakdown
    st.subheader("Current 1H Regime Features (from generate_features)")
    feats = data.get("features", {})
    cols = st.columns(4)
    for i, (k, v) in enumerate(feats.items()):
        cols[i % 4].metric(k, f"{v}")

    # 4. History Table
    st.markdown("---")
    st.subheader("Recent 1H Bar Classifications")
    hist = data.get("history", [])
    if hist:
        st.dataframe(pd.DataFrame(hist)[::-1], use_container_width=True)
    else:
        st.caption("Waiting for live 1H bar close to append to history...")

time.sleep(3)
st.rerun()
from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager
from ib_async import IB,Forex,util
import numpy as np
import pandas as pd
from pathlib import Path
from features.feature_eng import generate_features
import joblib

TWS_HOST="127.0.0.1"
TWS_PORT="7497"
CLIENT_ID=25
RVOL_THRESHOLD=1.20
SYMBOL = "EURUSD"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
model_path=(
    BASE_DIR
    / "src"
    / "models"
    / "trained"
    / "xgboost"
    / "eur_regime_rf.pkl"
)

ib=IB()
REGIME_FEATURES = [
    "ret_1",
    "ret_4",
    "ret_16",
    "volatility_16",
    "volatility_64",
    "range_pct",
    "body_pct",
    "abs_return",
]

model_artifact=None
contract=Forex(SYMBOL)

def classify_phase(rvol,range_pct,wick_ratio,body_pct):
    if rvol < 1.0 and range_pct < 0.0025:
        return "ACCUMULATION (Chop / Low Liquidity)"
    if wick_ratio > 0.55 and range_pct > 0.0030:
        return "MANIPULATION (Liquidity Sweep/Fakeout)"
    if rvol >= RVOL_THRESHOLD and body_pct > 0.45:
        return "DISTRIBUTION (Institutional Expansion)"
    return "CONTINUATION / TRANSITION"

state = {
    "connected": False,
    "last_bar_time": "Waiting...",
    "spot_price": 0.0,
    "current_session": "None",
    "market_phase": "INITIALIZING",
    "predicted_target": 0.0,
    "features": {feat: 0.0 for feat in REGIME_FEATURES},
    "history": [],
}
def on_bar_update(bars,has_new_bar):
    if not has_new_bar or len(bars)<33:
        return

    df=util.df(bars)

    feats_df = generate_features(df,horizon=4, threshold=0.001)

    latest_row=feats_df.iloc[-1]
    price= float(latest_row["close"])

    state["spot_price"]=price
    state["last_bar_time"]=str(df["date"].iloc[-1])

    # 1h morphology
    c_h=float(latest_row["high"])
    c_l=float(latest_row["low"])
    c_o=float(latest_row["open"])
    bar_range=max(c_h-c_l,1e-6)
    body_size=abs(price- c_o)

    range_pct=float(latest_row.get("range_pct",bar_range/price))
    body_pct=float(latest_row.get("body_pct",body_size/bar_range))
    wick_ratio=(bar_range-body_size)/bar_range

    for feat in REGIME_FEATURES:
        if feat in latest_row:
            state["features"][feat]=round(float(latest_row[feat]),5)

    # prediction
    pred_val=0.0
    if model_artifact:
        model=(model_artifact["model"]
               if isinstance(model_artifact,dict)
               else model_artifact
               )
        x_live=feats_df[REGIME_FEATURES].iloc[[-1]]
        x_live=x_live.replace([np.inf, -np.inf],np.nan).fillna(0)

        pred_val = float(model.predict(x_live)[0])
        state["predicted_target"]=round(pred_val,4)

    phase=classify_phase(pred_val,range_pct,body_pct,wick_ratio)
    state["market_phase"]=phase

    log_entry = {
        "Bar Time": state["last_bar_time"][-8:],
        "Price": f"{price:.5f}",
        "Phase": phase.split(" ")[0],
        "Predicted Target": round(pred_val, 4),
        "Range %": f"{range_pct * 100:.3f}%",
        "Body %": f"{body_pct * 100:.1f}%",
    }
    if (
        not state["history"]
        or state["history"][-1]["Bar Time"] != log_entry["Bar Time"]
    ):
        state["history"].append(log_entry)
        if len(state["history"]) > 10:
            state["history"].pop(0)

@asynccontextmanager
async def lifespan(app:FastAPI):
    global model_artifact
    if model_path.exists():
        model_artifact=joblib.load(model_path)
        print(f" Loaded trained 1H model from {model_path}")

    try:
        await ib.connectAsync(TWS_HOST,TWS_PORT,clientId=CLIENT_ID,timeout=10)
        state["connected"]=True
        await ib.qualifyContractsAsync(contract)

        bars=await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="10 D",
            barSizeSetting="1 hour",
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=1,
            keepUpToDate=True
        )

        bars.updateEvent += on_bar_update
        if len(bars) >= 70:
            on_bar_update(bars, has_new_bar=True)
        print("connected to IBKR. subscribed to live EUR/USD 1H bars")
    except Exception as e:
        print(f"connection error {e}")
        state["connected"]=False

    yield

    if ib.isConnected():
        ib.disconnect()

app=FastAPI(title="1H Market Phae Engine",lifespan=lifespan)

@app.get("/state")
def get_state():
    state["connected"]=ib.isConnected()
    return state





    

# 📊 EUR/USD Institutional Market Phase Engine (AMD)

An end-to-end Machine Learning and quantitative trading system for real-time institutional market phase classification on **EUR/USD**. The system integrates live 1-hour bar streaming from **Interactive Brokers (IBKR)**, on-the-fly feature engineering, a trained **XGBoost Random Forest** regime model, a **FastAPI** backend service, and a real-time **Streamlit** dashboard.

---

## 🏛️ Concept: AMD (Accumulation - Manipulation - Distribution)

The engine identifies smart money cycles based on candle morphology, volatility regimes, and machine learning predictions:

- **🔵 Accumulation (Chop / Low Liquidity)**: Narrow range compression, low volatility, institutional positioning.
- **🟠 Manipulation (Liquidity Sweep / Fakeout)**: High wick ratios rejecting key levels, sweeping stop orders.
- **🟢 Distribution (Institutional Expansion)**: High displacement, wide body percentages, expanded relative volume and volatility.

---

## ⚙️ Architecture & Data Flow

```mermaid
flowchart LR
    A["Interactive Brokers (TWS / Gateway)<br/>Port 7497"] -->|Live 1H EUR/USD Bars| B["FastAPI Backend<br/>(src/api/main.py)"]
    B -->|Bar Stream| C["Feature Engineering<br/>(src/features/feature_eng.py)"]
    C -->|Regime Features| D["XGBoost Model<br/>(src/models/trained/xgboost)"]
    D -->|Target Prediction| E["Phase Classification Logic<br/>(Accumulation / Manipulation / Distribution)"]
    E -->|JSON State /state| B
    B -->|HTTP Polling (Port 8000)| F["Streamlit Dashboard<br/>(src/dashboard/app.py)"]
```

---

## 🚀 Live System Output

### 1. Backend REST API (`GET http://127.0.0.1:8000/state`)

Live JSON snapshot returned by the running engine:

```json
{
  "connected": true,
  "last_bar_time": "2026-10-30 08:00:00-04:00",
  "spot_price": 1.1195,
  "current_session": "None",
  "market_phase": "ACCUMULATION (Chop / Low Liquidity)",
  "predicted_target": 0.6970,
  "features": {
    "ret_1": -0.00004,
    "ret_4": 0.00004,
    "ret_16": 0.00134,
    "volatility_16": 0.00061,
    "volatility_64": 0.00066,
    "range_pct": 0.00009,
    "body_pct": -0.00004,
    "abs_return": 0.00004
  },
  "history": [
    {
      "Bar Time": "00-04:00",
      "Price": "1.12010",
      "Phase": "ACCUMULATION",
      "Predicted Target": 0.7167,
      "Range %": "0.067%",
      "Body %": "0.0%"
    }
  ]
}
```

### 2. Streamlit Interface (`http://127.0.0.1:8501`)

The frontend renders real-time quantitative monitoring cards:
- **Top Metrics**:
  - `IBKR Socket`: `CONNECTED` (Green)
  - `EUR/USD Spot`: `1.11950`
  - `Last 1H Bar`: `2026-10-30 08:00:00-04:00`
  - `Model Prediction`: `0.6970`
- **Institutional Phase Banner**:
  - `🔵 Phase: ACCUMULATION (Chop / Low Liquidity)`
  - *Caption: Market is coiling. Narrow candle range and contracting volatility.*
- **Current 1H Regime Features**:
  - Displays dynamic cards for `ret_1`, `ret_4`, `ret_16`, `volatility_16`, `volatility_64`, `range_pct`, `body_pct`, and `abs_return`.
- **Classification History Table**:
  - Tabular log tracking recent 1H bar close evaluations, price, classified phase, target score, and candle metrics.

---

## 📁 Repository Structure

```text
├── .gitignore
├── README.md
└── src/
    ├── requirements-full.txt
    ├── api/
    │   ├── __init__.py
    │   └── main.py              # FastAPI server & IBKR event loop
    ├── dashboard/
    │   └── app.py               # Streamlit real-time monitoring interface
    ├── features/
    │   └── feature_eng.py       # Rolling volatility, returns & morphology engineering
    └── models/
        ├── xg.py                # Model training pipeline
        └── trained/
            └── xgboost/         # Serialized model weights (*.pkl)
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- Interactive Brokers Trader Workstation (TWS) or IB Gateway running with API port enabled (`7497` for paper trading or `7496` for live).

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/peaceboii/EURUSD-AMD.git
cd EURUSD-AMD

# Create and activate virtual environment
python -m venv market_ml
.\market_ml\Scripts\Activate.ps1

# Install dependencies
pip install -r src/requirements-full.txt
```

### 3. Run the Backend API
```powershell
cd src
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 4. Run the Streamlit Dashboard
In a separate terminal:
```powershell
cd src
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

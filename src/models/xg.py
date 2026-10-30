import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,TimeSeriesSplit
import pandas as pd
import numpy as np
from features.feature_eng import generate_features
from sklearn.metrics import mean_absolute_error,mean_squared_error
eur_data=pd.read_csv("C:/Users/kv/Programming/dsa/Interview-Mastery/15_ML&DL_concepts/aiml_projects/global-market-ml/src/data/raw_data/EUR_1H.csv")
aapl_data=pd.read_csv("C:/Users/kv/Programming/dsa/Interview-Mastery/15_ML&DL_concepts/aiml_projects/global-market-ml/src/data/raw_data/AAPL_1H.csv")

eur_df=generate_features(eur_data,5,0.5)
aapl_df=generate_features(aapl_data,5,0.1)



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

eur_df=eur_df.dropna(subset=REGIME_FEATURES+["target"])
aapl_df=aapl_df.dropna(subset=REGIME_FEATURES+["target"])

eur_X,eur_y=eur_df[REGIME_FEATURES],eur_df["target"]
aapl_X,aapl_y=aapl_df[REGIME_FEATURES],aapl_df["target"]

def train(df,name):
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=REGIME_FEATURES + ["target"]
    )

    X=df[REGIME_FEATURES]
    y=df["target"]
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=False)

    model=XGBRegressor(n_estimators=1,num_parallel_tree=100,max_depth=5,subsample=0.8,colsample_bynode=0.8,random_state=42)
    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(name)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    print("\nFeature importance:")
    importances = pd.Series(
        model.feature_importances_, index=REGIME_FEATURES
    ).sort_values(ascending=False)
    for feat, val in importances.items():
        print(f"  {feat:15s}: {val:.4f}")

    models_dir=Path("models/trained/xgboost")
    models_dir.mkdir(parents=True,exist_ok=True)

    save_path=models_dir/f"{name.lower()}_regime_rf.pkl"
    joblib.dump({"model":model,"features":REGIME_FEATURES},save_path)
    print(f"model saved: {save_path}")

train(eur_df,name="EUR")
train(aapl_df,name="AAPL")



import pandas as pd
import numpy as np
import os

os.makedirs("feats_data",exist_ok=True)

import pandas as pd

def prepare_timestamps(path):
    df = pd.read_csv(path).copy()

    # Parse timestamps and normalize them to UTC.
    df["date"] = pd.to_datetime(
        df["date"],
        utc=True
    )

    # Sort chronologically and create an independent DataFrame.
    df = (
        df.sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    # Calculate elapsed time between bars.
    df.loc[:, "gap"] = df["date"].diff()

    return df

def generate_features(df,horizon,threshold):
    data=df.copy()

    data["ret_1"]=data["close"].pct_change(1)
    data["ret_4"]=data["close"].pct_change(4)
    data["ret_16"]=data["close"].pct_change(16)

    data["log_return"]=np.log(data["close"]/data["close"].shift(1))

    data["volatility_16"] =data["log_return"].rolling(16).std()
    data["volatility_64"] = data["log_return"].rolling(64).std()

    data["range_pct"]=((data["high"]-data["low"])/data["open"]).clip(lower=1e-6)
    data["body_pct"]=(data["close"]-data["open"])/data["open"]

    data["abs_return"]=data["log_return"].abs()

    # target
    fwd_vol=(data["log_return"].iloc[::-1].rolling(window=horizon).std().iloc[::-1])

    baseline_vol=data["volatility_64"].rolling(64,min_periods=16).median()

    data["target"] = (fwd_vol > baseline_vol * threshold).astype(int) 



    return data

if __name__ == "__main__":
    eur = prepare_timestamps("C:/Users/kv/Programming/dsa/Interview-Mastery/15_ML&DL_concepts/aiml_projects/global-market-ml/src/data/raw_data/EUR_1H.csv")
    aapl = prepare_timestamps("C:/Users/kv/Programming/dsa/Interview-Mastery/15_ML&DL_concepts/aiml_projects/global-market-ml/src/data/raw_data/AAPL_1H.csv")

    eur_csv=pd.read_csv("C:/Users/kv/Programming/dsa/Interview-Mastery/15_ML&DL_concepts/aiml_projects/global-market-ml/src/data/raw_data/EUR_1H.csv")
    aapl_csv=pd.read_csv("C:/Users/kv/Programming/dsa/Interview-Mastery/15_ML&DL_concepts/aiml_projects/global-market-ml/src/data/raw_data/AAPL_1H.csv")

    Eur_feats_df=generate_features(eur,16,1.15)
    Aapl_feats_df=generate_features(aapl,16,1.15)


    Eur_feats_df.to_csv(f"feats_data/Eur_feats.csv")
    Aapl_feats_df.to_csv(f"feats_data/Aapl_feats.csv")

    print(Eur_feats_df.tail(10))

    print(Eur_feats_df.isna().sum())
    timestamps = pd.to_datetime(
        Eur_feats_df["date"],
        utc=True,
    )

    print(
        timestamps.diff()
        .value_counts()
        .head(10)
    )


    print(Aapl_feats_df.isna().sum())
    timestamps = pd.to_datetime(
        Aapl_feats_df["date"],
        utc=True,
    )

    print(
        timestamps.diff()
        .value_counts()
        .head(10)
    )
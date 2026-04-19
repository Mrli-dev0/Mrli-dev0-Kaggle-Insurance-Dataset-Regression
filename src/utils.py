import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import random
import faiss

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

SEED = 42
N_FOLDS = 10
N_OPTUNA_TRIALS = 10
CLIP_PERCENTILE = [1, 99]
TARGET_NAME = "Premium Amount"

def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def check_lgb_gpu():
    try:
        lgb.cuda_is_available()
        return True
    except:
        return False

def faiss_knn_imputer_gpu(X, n_neighbors=5):
    X = X.astype(np.float32)
    non_nan_mask = ~np.isnan(X).any(axis=1)
    X_non_nan = X[non_nan_mask]
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X_non_nan)
    X_imputed = X.copy()
    nan_rows = np.where(np.isnan(X).any(axis=1))[0]
    for i in nan_rows:
        query = np.nan_to_num(X[i]).reshape(1, -1)
        D, I = index.search(query, n_neighbors)
        neighbors = X[non_nan_mask][I[0]]
        X_imputed[i] = np.nanmean(neighbors, axis=0)
    return X_imputed

def rmsle(y_true, y_pred):
    y_true = np.clip(y_true, 1e-6, None)
    y_pred = np.clip(y_pred, 1e-6, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def get_dynamic_weights(scores):
    scores = np.array(scores)
    inv = 1 / (scores + 1e-8)
    return inv / inv.sum()

def feature_engineering(train_path, test_path):
    train_pl = pl.read_csv(train_path)
    test_pl = pl.read_csv(test_path)
    target = TARGET_NAME
    y = train_pl.select(target).to_pandas().values.ravel()

    train_pl = train_pl.rename({c: c.strip().lower().replace(' ', '_') for c in train_pl.columns})
    test_pl = test_pl.rename({c: c.strip().lower().replace(' ', '_') for c in test_pl.columns})
    target = target.lower().replace(' ', '_')

    train_feats = train_pl.drop(target)
    test_feats = test_pl
    df = pl.concat([train_feats, test_feats], how="vertical")

    for col in df.columns:
        if df[col].dtype in [pl.String, pl.Boolean]:
            df = df.with_columns(pl.col(col).fill_null("unknown"))
        else:
            if col != "id":
                df = df.with_columns(pl.col(col).fill_null(pl.col(col).median()).fill_null(0))

    df = df.with_columns([
        (pl.col("age") * pl.col("vehicle_age")).alias("age_vehicleage"),
        (pl.col("annual_income") * pl.col("health_score")).alias("income_health"),
        (pl.col("credit_score") * pl.col("insurance_duration")).alias("credit_duration"),
        (pl.col("age") * pl.col("credit_score")).alias("age_credit"),
        (pl.col("health_score") / (pl.col("age") + 1)).alias("health_per_age"),
        (pl.col("vehicle_age") / (pl.col("age") + 1)).alias("vehicle_per_age"),
        (pl.col("credit_score") / (pl.col("annual_income") + 1e-8)).alias("credit_per_income"),
        (pl.col("insurance_duration") / (pl.col("vehicle_age") + 1e-8)).alias("duration_per_vehicle"),
        (pl.col("annual_income") / (pl.col("age") + 1)).alias("income_per_age"),
        (pl.col("credit_score") / (pl.col("health_score") + 1e-8)).alias("credit_per_health"),
        (pl.col("insurance_duration") / (pl.col("age") + 1)).alias("duration_per_age"),
        (pl.col("annual_income").log1p()).alias("log_income"),
        (pl.col("age") * pl.col("health_score")).alias("age_health"),
    ])

    for c in ["location", "occupation", "marital_status", "education", "policy_type"]:
        if c in df.columns:
            cnt = df.group_by(c).len().to_dicts()
            cnt_map = {d[c]: d["len"] for d in cnt}
            df = df.with_columns(pl.col(c).replace(cnt_map, default=0).alias(f"{c}_cnt"))

    if "policy_start_date" in df.columns:
        df = df.with_columns(pl.col("policy_start_date").str.to_datetime(strict=False))
        df = df.with_columns([
            pl.col("policy_start_date").dt.year().fill_null(0).alias("policy_year"),
            pl.col("policy_start_date").dt.month().fill_null(0).alias("policy_month"),
            pl.col("policy_start_date").dt.quarter().fill_null(0).alias("policy_quarter"),
        ])

    for c in df.columns:
        if df[c].dtype == pl.String:
            df = df.with_columns(pl.col(c).cast(pl.Categorical).to_physical())

    X_train = df.head(len(train_feats)).to_pandas()
    X_test = df.tail(len(test_feats)).to_pandas()

    cat_feats = [c for c in X_train.columns if X_train[c].nunique() < 50 and c != "id"]
    te = TargetEncoder(cols=cat_feats, smoothing=10)
    X_train = pd.concat([X_train, te.fit_transform(X_train[cat_feats], y).add_prefix("te_")], axis=1)
    X_test = pd.concat([X_test, te.transform(X_test[cat_feats]).add_prefix("te_")], axis=1)

    drop_cols = ["id", "policy_start_date"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns], errors="ignore")
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], errors="ignore")
    X_train = X_train.fillna(0).astype(float)
    X_test = X_test.fillna(0).astype(float)

    vt = VarianceThreshold(0.01)
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)

    mi = mutual_info_regression(X_train, y, random_state=SEED, n_jobs=-1)
    mask = mi >= np.percentile(mi, 10)
    X_train = X_train[:, mask]
    X_test = X_test[:, mask]

    return X_train, X_test, y, te, vt, mask
import os
import warnings
import joblib
import optuna
import numpy as np
from utils import *

warnings.filterwarnings('ignore')
seed_everything()

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

def objective(trial):
    X, _, y, _, _, _ = feature_engineering(TRAIN_PATH, TEST_PATH)

    y_original = y.copy()
    y = np.clip(y, *np.percentile(y, [1, 99]))
    y_log = np.log1p(y)

    kf = KFold(5, shuffle=True, random_state=SEED)

    params = {
        "lgb": {
            "learning_rate": trial.suggest_float("lgb_learning_rate", 0.05, 0.15),
            "max_depth": trial.suggest_int("lgb_max_depth", 3, 6),
            "reg_alpha": trial.suggest_float("lgb_reg_alpha", 0.01, 2),
            "reg_lambda": trial.suggest_float("lgb_reg_lambda", 0.01, 2),
            "colsample_bytree": 0.7,
            "subsample": 0.7,
            "n_estimators": 300,
            "random_state": SEED,
            "verbose": -1
        },
        "xgb": {
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.05, 0.15),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 6),
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "colsample_bytree": 0.7,
            "subsample": 0.7,
            "eval_metric": "rmse",
            "n_estimators": 300,
            "random_state": SEED
        },
        "cat": {
            "learning_rate": trial.suggest_float("cat_learning_rate", 0.05, 0.12),
            "depth": trial.suggest_int("cat_depth", 3, 5),
            "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1, 5),
            "n_estimators": 300,
            "random_state": SEED
        }
    }

    scores, model_scores = [], []
    for tr, val in kf.split(X):
        xt, xv = X[tr], X[val]
        yt, yv = y_log[tr], y_log[val]
        yv_real = y_original[val]

        scaler = StandardScaler()
        xs_t = scaler.fit_transform(xt)
        xs_v = scaler.transform(xv)

        m1 = lgb.LGBMRegressor(**params["lgb"], device="cuda", gpu_use_dp=False)
        m1.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(50, verbose=0)])

        m2 = xgb.XGBRegressor(**params["xgb"], tree_method="hist", device="cuda")
        m2.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)

        m3 = CatBoostRegressor(**params["cat"], task_type="GPU", verbose=0)
        m3.fit(xt, yt, eval_set=(xv, yv), use_best_model=True)

        m4 = lgb.LGBMRegressor(boosting_type="rf", n_estimators=100, max_depth=6, bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, device="cuda", random_state=SEED, verbose=-1)
        m4.fit(xt, yt)

        m5 = ExtraTreesRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=SEED)
        m5.fit(xs_t, yt)

        m6 = Ridge(alpha=1.0, random_state=SEED)
        m6.fit(xs_t, yt)

        p1, p2, p3, p4, p5, p6 = m1.predict(xv), m2.predict(xv), m3.predict(xv), m4.predict(xv), m5.predict(xv), m6.predict(xv)

        s = [rmsle(yv_real, np.expm1(p)) for p in [p1,p2,p3,p4,p5,p6]]
        model_scores.append(s)
        w = get_dynamic_weights(s)

        final_log = np.average([p1,p2,p3,p4,p5,p6], weights=w, axis=0)
        final = np.expm1(final_log)

        scores.append(rmsle(yv_real, final))

    print("📊 单模型 RMSLE:", np.round(np.mean(model_scores, axis=0), 4))
    return np.mean(scores)

def train_full():
    X, Xt, y, te, vt, mask = feature_engineering(TRAIN_PATH, TEST_PATH)
    X = faiss_knn_imputer_gpu(X, 5)

    y = np.clip(y, *CLIP_PERCENTILE)
    y_log = np.log1p(y)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)
    best = study.best_params

    best_params = {
        "lgb": {
            "learning_rate": best["lgb_learning_rate"],
            "max_depth": best["lgb_max_depth"],
            "reg_alpha": best["lgb_reg_alpha"],
            "reg_lambda": best["lgb_reg_lambda"],
            "colsample_bytree": 0.7,
            "subsample": 0.7,
            "device": "cuda",
            "gpu_use_dp": False,
            "n_estimators": 2000,
            "random_state": SEED,
            "verbose": -1
        },
        "xgb": {
            "learning_rate": best["xgb_learning_rate"],
            "max_depth": best["xgb_max_depth"],
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "colsample_bytree": 0.7,
            "subsample": 0.7,
            "tree_method": "hist",
            "device": "cuda",
            "eval_metric": "rmse",
            "random_state": SEED
        },
        "cat": {
            "learning_rate": best["cat_learning_rate"],
            "depth": best["cat_depth"],
            "l2_leaf_reg": best["cat_l2_leaf_reg"],
            "task_type": "GPU",
            "random_state": SEED,
            "verbose": 0
        }
    }

    joblib.dump(best_params, os.path.join(MODEL_DIR, "best_params.pkl"))
    joblib.dump(te, os.path.join(MODEL_DIR, "target_encoder.pkl"))
    joblib.dump(vt, os.path.join(MODEL_DIR, "vt.pkl"))
    joblib.dump(mask, os.path.join(MODEL_DIR, "mask.pkl"))
    joblib.dump(y, os.path.join(MODEL_DIR, "y_clip.pkl"))

    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    for fold, (tr, val) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{N_FOLDS}")
        xt, xv = X[tr], X[val]
        yt, yv = y_log[tr], y_log[val]
        xs_t = X_s[tr]

        m1 = lgb.LGBMRegressor(**best_params["lgb"])
        m1.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(200, verbose=0)])

        m2 = xgb.XGBRegressor(**best_params["xgb"])
        m2.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)

        m3 = CatBoostRegressor(**best_params["cat"])
        m3.fit(xt, yt, eval_set=(xv, yv), use_best_model=True)

        m4 = lgb.LGBMRegressor(boosting_type="rf", bagging_freq=1, bagging_fraction=0.8, device="cuda", random_state=SEED, verbose=-1)
        m4.fit(xt, yt)

        m5 = ExtraTreesRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=SEED)
        m5.fit(xs_t, yt)

        m6 = Ridge(alpha=1.0, random_state=SEED)
        m6.fit(xs_t, yt)

        joblib.dump(m1, os.path.join(MODEL_DIR, f"lgb_{fold}.pkl"))
        joblib.dump(m2, os.path.join(MODEL_DIR, f"xgb_{fold}.pkl"))
        joblib.dump(m3, os.path.join(MODEL_DIR, f"cat_{fold}.pkl"))
        joblib.dump(m4, os.path.join(MODEL_DIR, f"rf_{fold}.pkl"))
        joblib.dump(m5, os.path.join(MODEL_DIR, f"et_{fold}.pkl"))
        joblib.dump(m6, os.path.join(MODEL_DIR, f"ridge_{fold}.pkl"))

    print("模型训练完成")

if __name__ == "__main__":
    train_full()
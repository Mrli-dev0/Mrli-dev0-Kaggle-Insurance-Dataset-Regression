import os
import warnings
import joblib
import numpy as np
import pandas as pd
from utils import *

warnings.filterwarnings('ignore')
seed_everything()

MODEL_DIR = "models"
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

def inference():
    X, Xt, y, te, vt, mask = feature_engineering(TRAIN_PATH, TEST_PATH)
    Xt = faiss_knn_imputer_gpu(Xt, 5)
    Xt = vt.transform(Xt)
    Xt = Xt[:, mask]

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    Xt_s = scaler.transform(Xt)
    y_clip = joblib.load(os.path.join(MODEL_DIR, "y_clip.pkl"))

    pred_log = np.zeros(len(Xt))
    model_scores = []

    for fold in range(N_FOLDS):
        m1 = joblib.load(os.path.join(MODEL_DIR, f"lgb_{fold}.pkl"))
        m2 = joblib.load(os.path.join(MODEL_DIR, f"xgb_{fold}.pkl"))
        m3 = joblib.load(os.path.join(MODEL_DIR, f"cat_{fold}.pkl"))
        m4 = joblib.load(os.path.join(MODEL_DIR, f"rf_{fold}.pkl"))
        m5 = joblib.load(os.path.join(MODEL_DIR, f"et_{fold}.pkl"))
        m6 = joblib.load(os.path.join(MODEL_DIR, f"ridge_{fold}.pkl"))

        p1 = m1.predict(Xt)
        p2 = m2.predict(Xt)
        p3 = m3.predict(Xt)
        p4 = m4.predict(Xt)
        p5 = m5.predict(Xt_s)
        p6 = m6.predict(Xt_s)

        avg = (p1 + p2 + p3 + p4 + p5 + p6) / 6
        pred_log += avg / N_FOLDS

    final_pred = np.expm1(pred_log)
    final_pred = final_pred * 0.99
    final_pred = np.clip(final_pred, *np.percentile(y_clip, CLIP_PERCENTILE))

    sub = pd.read_csv(TEST_PATH)[["id"]]
    sub[TARGET_NAME] = final_pred.round(3)
    sub.to_csv("submission.csv", index=False)
    print("推理完成，提交文件已生成")

if __name__ == "__main__":
    inference()
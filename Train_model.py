# train_models.py
import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

RANDOM_STATE = 42

def clean_text(text):
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    # 1) load datasets (expects Fake.csv & True.csv in working dir)
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    df_fake["label"] = 1
    df_true["label"] = 0

    df = pd.concat([df_fake[['text','label']], df_true[['text','label']]], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # cleaning
    df['clean'] = df['text'].apply(clean_text)

    X = df['clean']
    y = df['label']

    # 2) split to (train, test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # 3) further split train -> subtrain + calibration (for SVM calibration)
    X_subtrain, X_cal, y_subtrain, y_cal = train_test_split(
        X_train, y_train, test_size=0.10, random_state=RANDOM_STATE, stratify=y_train
    )

    # 4) TF-IDF: fit on subtrain (we will use same TF-IDF for SVM components)
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_subtrain_vec = tfidf.fit_transform(X_subtrain)
    X_cal_vec = tfidf.transform(X_cal)
    X_test_vec = tfidf.transform(X_test)

    # 5) Oversample subtrain vectors to balance classes
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_subtrain_res, y_subtrain_res = ros.fit_resample(X_subtrain_vec, y_subtrain)

    # 6) Train LinearSVC on resampled data
    svc = LinearSVC(max_iter=5000, random_state=RANDOM_STATE)
    svc.fit(X_subtrain_res, y_subtrain_res)

    # 7) Calibrate SVM (so we can get probabilities)
    # Use CalibratedClassifierCV with cv='prefit' to calibrate using X_cal_vec
    calib_svc = CalibratedClassifierCV(svc, cv='prefit')
    calib_svc.fit(X_cal_vec, y_cal)

    # 8) Train Logistic Regression pipeline (with its own TF-IDF, for simplicity we'll use a separate pipeline)
    # Fit Logistic on full X_train (not only subtrain)
    tfidf_log = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_train_vec_log = tfidf_log.fit_transform(X_train)
    X_test_vec_log = tfidf_log.transform(X_test)

    log_model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)
    log_model.fit(X_train_vec_log, y_train)

    # 9) Evaluate both models on test set
    print("=== Evaluation on TEST set ===")
    # SVM (calibrated) using tfidf from SVM pipeline (tfidf)
    y_pred_svm = calib_svc.predict(X_test_vec)
    if hasattr(calib_svc, "predict_proba"):
        y_proba_svm = calib_svc.predict_proba(X_test_vec)[:,1]
        try:
            roc_svm = roc_auc_score(y_test, y_proba_svm)
        except:
            roc_svm = None
    else:
        y_proba_svm = None
        roc_svm = None

    print("\n-- Calibrated SVM --")
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))
    if roc_svm is not None:
        print("ROC AUC:", roc_svm)
    print(classification_report(y_test, y_pred_svm))

    # Logistic
    y_pred_log = log_model.predict(X_test_vec_log)
    y_proba_log = log_model.predict_proba(X_test_vec_log)[:,1]
    roc_log = roc_auc_score(y_test, y_proba_log)

    print("\n-- Logistic Regression --")
    print("Accuracy:", accuracy_score(y_test, y_pred_log))
    print("ROC AUC:", roc_log)
    print(classification_report(y_test, y_pred_log))

    # 10) Save artifacts
    # Save Logistic pipeline as single file (tfidf + logistic) using joblib
    from sklearn.pipeline import Pipeline
    log_pipeline = Pipeline([
        ('tfidf', tfidf_log),
        ('clf', log_model)
    ])
    joblib.dump(log_pipeline, "logreg_pipeline.pkl")
    print("Saved: logreg_pipeline.pkl")

    # Save SVM components (tfidf used for SVM + calibrated classifier)
    joblib.dump(tfidf, "svm_tfidf.pkl")
    joblib.dump(calib_svc, "svm_calibrated.pkl")
    print("Saved: svm_tfidf.pkl, svm_calibrated.pkl")

    # Also save a default model file for Flask (choose logistic by default since it has smooth probabilities)
    joblib.dump(log_pipeline, "model_hoax_pipeline.pkl")
    print("Saved: model_hoax_pipeline.pkl (default for Flask)")

if __name__ == "__main__":
    main()


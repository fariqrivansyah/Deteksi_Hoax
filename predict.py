# predict.py
import joblib
import sys
import re

def clean_text(text):
    s = str(text).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_models():
    # default logistic pipeline
    log_pipe = joblib.load("logreg_pipeline.pkl")
    # SVM components
    try:
        svm_tfidf = joblib.load("svm_tfidf.pkl")
        svm_clf = joblib.load("svm_calibrated.pkl")
    except:
        svm_tfidf, svm_clf = None, None
    return log_pipe, svm_tfidf, svm_clf

def predict_with_logistic(pipe, text):
    pred = pipe.predict([text])[0]
    prob = pipe.predict_proba([text])[0][1]
    return pred, prob

def predict_with_svm(tfidf, clf, text):
    vec = tfidf.transform([text])
    pred = clf.predict(vec)[0]
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(vec)[0][1]
    else:
        prob = None
    return pred, prob

if __name__ == "__main__":
    log_pipe, svm_tfidf, svm_clf = load_models()
    if len(sys.argv) > 1:
        raw = " ".join(sys.argv[1:])
    else:
        raw = input("Masukkan teks berita untuk prediksi: ")

    cleaned = clean_text(raw)

    print("\n== Logistic Regression ==")
    p_log, prob_log = predict_with_logistic(log_pipe, cleaned)
    print("Label:", "HOAX" if p_log==1 else "VALID")
    print("Prob(HOAX):", round(prob_log,3))

    if svm_tfidf is not None and svm_clf is not None:
        print("\n== SVM (calibrated) ==")
        p_svm, prob_svm = predict_with_svm(svm_tfidf, svm_clf, cleaned)
        print("Label:", "HOAX" if p_svm==1 else "VALID")
        if prob_svm is not None:
            print("Prob(HOAX):", round(prob_svm,3))
    else:
        print("\nSVM model components not found.")

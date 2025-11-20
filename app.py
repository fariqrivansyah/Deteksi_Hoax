# app.py
from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load default model (Logistic pipeline)
log_pipe = joblib.load("logreg_pipeline.pkl")
log_pipe = joblib.load("model_hoax_pipeline.pkl")

# Load SVM components if present
try:
    svm_tfidf = joblib.load("svm_tfidf.pkl")
    svm_clf = joblib.load("svm_calibrated.pkl")
except:
    svm_tfidf = None
    svm_clf = None

def clean_text(text):
    s = str(text).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None
    chosen = "logistic"
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("text", "")
        chosen = request.form.get("model", "logistic")
        cleaned = clean_text(input_text)

        if chosen == "logistic":
            pred = log_pipe.predict([cleaned])[0]
            prob = float(log_pipe.predict_proba([cleaned])[0][1])
            result = "HOAX" if pred==1 else "VALID"
        else:
            if svm_tfidf is not None and svm_clf is not None:
                vec = svm_tfidf.transform([cleaned])
                pred = svm_clf.predict(vec)[0]
                result = "HOAX" if pred==1 else "VALID"
                prob = None
                if hasattr(svm_clf, "predict_proba"):
                    prob = float(svm_clf.predict_proba(vec)[0][1])
            else:
                result = "SVM model not available"

    return render_template("index.html", result=result, prob=prob, chosen=chosen, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)

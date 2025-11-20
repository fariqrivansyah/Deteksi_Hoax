from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import re
import joblib

# ===================
# LOAD DATASET
# ===================
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake["label"] = 1
df_true["label"] = 0

df = pd.concat([
    df_fake[["text", "label"]],
    df_true[["text", "label"]]
], ignore_index=True)


# ===================
# CLEAN TEXT
# ===================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean"] = df["text"].apply(clean_text)

X = df["clean"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ===================
# PIPELINE STABIL
# ===================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("oversample", RandomOverSampler()),
    ("svm", LinearSVC())
])

# ===================
# TRAIN
# ===================
pipeline.fit(X_train, y_train)

# ===================
# EVALUATION
# ===================
y_pred = pipeline.predict(X_test)
print("\n=== Evaluation ===")
print(classification_report(y_test, y_pred))

# ===================
# SAVE MODEL
# ===================
joblib.dump(pipeline, "model_hoax_pipeline.pkl")

print("\nModel saved as model_hoax_pipeline.pkl")

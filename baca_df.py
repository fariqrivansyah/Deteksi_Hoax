import joblib
import pandas as pd

# 1. Load pipeline model
pipeline = joblib.load("model_hoax_pipeline.pk")

# 2. Baca dataset untuk uji coba
df = pd.read_csv("sample_test.csv")       # ganti sesuai nama file kamu
X_test = df['text']                       # kolom teks yang mau diuji

# 3. Prediksi
y_pred = pipeline.predict(X_test)

# 4. Tampilkan hasil
for teks, pred in zip(X_test, y_pred):
    print("TEKS:", teks)
    print("PREDIKSI:", pred)
    print("-" * 60)





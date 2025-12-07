# test_and_save.py
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
STAND_SCALER_PATH = os.path.join(BASE_DIR, 'standscaler.pkl')
MINMAX_SCALER_PATH = os.path.join(BASE_DIR, 'minmaxscaler.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

# --- Load dataset ---
df = pd.read_csv('Crop_recommendation.csv')

# Adjust these column names if your CSV uses different names:
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Optional: inspect columns to confirm (uncomment for debugging)
# print("X columns:", X.columns)
# print("Unique labels sample:", y.unique()[:10])

# Encode string labels to integers if necessary
le = None
if y.dtype == object or y.dtype.name == 'category':
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")
else:
    y_enc = y

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Scalers: keep the same order you use in app (MinMax then Standard)
ms = MinMaxScaler().fit(X_train)
X_train_min = ms.transform(X_train)

sc = StandardScaler().fit(X_train_min)
X_train_final = sc.transform(X_train_min)

# Train model on scaled features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_final, y_train)

# Save model & scalers with joblib (preferred for sklearn objects)
joblib.dump(model, MODEL_PATH)
joblib.dump(sc, STAND_SCALER_PATH)
joblib.dump(ms, MINMAX_SCALER_PATH)

print("Saved model and scalers with joblib:")
print(" -", MODEL_PATH)
print(" -", STAND_SCALER_PATH)
print(" -", MINMAX_SCALER_PATH)

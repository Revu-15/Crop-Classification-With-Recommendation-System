import streamlit as st
import numpy as np
import joblib
import os
from pathlib import Path

st.set_page_config(page_title="Crop Recommendation", layout="centered")

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
STAND_SCALER_PATH = os.path.join(BASE_DIR, "standscaler.pkl")
MINMAX_SCALER_PATH = os.path.join(BASE_DIR, "minmaxscaler.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
IMAGE_PATH = os.path.join(BASE_DIR, "static/img.jpg")

# ---------------- Loader ----------------
@st.cache_resource
def safe_load(path):
    try:
        p = Path(path)
        if p.exists():
            return joblib.load(path)
        else:
            return None
    except Exception as e:
        st.warning(f"Failed to load {os.path.basename(path)}: {e}")
        return None

model = safe_load(MODEL_PATH)
sc = safe_load(STAND_SCALER_PATH)
ms = safe_load(MINMAX_SCALER_PATH)
label_encoder = safe_load(LABEL_ENCODER_PATH)

# ---------------- UI ----------------
st.markdown(
    """
    <h1 style='text-align:center; color:mediumseagreen;'>
        Crop Recommendation System 🌱
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Enter all the values below and click **Get Recommendation**.")

# Warning if files missing
if model is None or sc is None or ms is None:
    st.error("Model or Scalers missing. Please upload all .pkl files.")
    st.write({
        "model.pkl": Path(MODEL_PATH).exists(),
        "standscaler.pkl": Path(STAND_SCALER_PATH).exists(),
        "minmaxscaler.pkl": Path(MINMAX_SCALER_PATH).exists(),
        "label_encoder.pkl": Path(LABEL_ENCODER_PATH).exists(),
    })

# ---------------- Input Form ----------------
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Nitrogen = st.number_input("Nitrogen", step=0.01)
    with col2:
        Phosporus = st.number_input("Phosphorus", step=0.01)
    with col3:
        Potassium = st.number_input("Potassium", step=0.01)

    col4, col5, col6 = st.columns(3)

    with col4:
        Temperature = st.number_input("Temperature (°C)", step=0.01)
    with col5:
        Humidity = st.number_input("Humidity (%)", step=0.01)
    with col6:
        Ph = st.number_input("pH", step=0.01)

    Rainfall = st.number_input("Rainfall (mm)", step=0.01)

    submit = st.form_submit_button("Get Recommendation")

# ---------------- Prediction ----------------
if submit:
    if model is None or sc is None or ms is None:
        st.error("Model or scaler not loaded. Fix files and restart.")
    else:
        values = np.array([[Nitrogen, Phosporus, Potassium, Temperature, Humidity, Ph, Rainfall]])

        try:
            scaled = ms.transform(values)
            final = sc.transform(scaled)
            pred = model.predict(final)
            pred0 = pred[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Decode output
        crop_name = None

        if label_encoder is not None:
            try:
                crop_name = label_encoder.inverse_transform([int(pred0)])[0]
            except:
                crop_name = None

        if crop_name is None and isinstance(pred0, str):
            crop_name = pred0

        if crop_name is None:
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
                6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
                10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
                17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
                20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }
            crop_name = crop_dict.get(int(pred0))

        # ---------------- Show Result ----------------
        st.success(f"🌱 Recommended Crop: **{crop_name}**")

        if Path(IMAGE_PATH).exists():
            st.image(IMAGE_PATH, caption="Crop Recommendation", width=300)
        else:
            st.info("Image not found: static/img.jpg")

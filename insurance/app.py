import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib


st.set_page_config(
    page_title="Insurance Prediction",
    page_icon="💰",
    layout="centered"
)


st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.header {
    background: linear-gradient(90deg, #0d6efd, #6610f2);
    padding: 1.2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.prediction {
    background: linear-gradient(90deg, #198754, #20c997);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
}
.big-number {
    font-size: 2.2rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="header">
    <h1>💰 Insurance Charges Prediction</h1>
    <p>ANN Model • Pretrained</p>
</div>
""", unsafe_allow_html=True)


# @st.cache_resource
# def load_objects():
#     model = tf.keras.models.load_model(
#         "insurance_model.h5",
#         compile=False
#     )
#     with open("scaler.pkl", "rb") as f:
#         scaler = joblib.load(f)
#     with open("columns.pkl", "rb") as f:
#         columns = joblib.load(f)
#     return model, scaler, columns


# model, scaler, columns = load_objects()
model= tf.keras.models.load_model("insurance_model.h5")
scaler=joblib.load("scaler.pkl")
columns= joblib.load("columns.pkl")


st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧑 User Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region",
        ["northwest", "northeast", "southwest", "southeast"]
    )

predict_btn = st.button("🔮 Predict Insurance Charges", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


if predict_btn:
    input_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    predicted_charge = prediction[0][0]

    st.markdown(f"""
    <div class="prediction">
        <h2>📊 Estimated Insurance Cost</h2>
        <div class="big-number">₹ {predicted_charge:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

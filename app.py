import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

import pandas as pd
import joblib

# ---------------------- LOAD FILES ----------------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ---------------------- CSS ----------------------
st.markdown("""
<style>

.main{
    background:#F8FAFC;
}

.block-container{
    padding-top:2rem;
}

h1{
    text-align:center;
    color:#C62828;
}

.stButton>button{
    width:100%;
    height:52px;
    border-radius:10px;
    background:linear-gradient(90deg,#E53935,#C62828);
    color:white;
    font-size:18px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    color:white;
}

.ok{
    background:#E8F5E9;
    padding:18px;
    border-radius:10px;
    color:#2E7D32;
    font-size:22px;
    font-weight:bold;
    text-align:center;
}

.bad{
    background:#FFEBEE;
    padding:18px;
    border-radius:10px;
    color:#C62828;
    font-size:22px;
    font-weight:bold;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------- TITLE ----------------------

st.markdown(
    "<h1>❤️ Heart Disease Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<center>Predict the likelihood of heart disease using a trained KNN model.</center>",
    unsafe_allow_html=True
)

st.write("")

# ---------------------- INPUTS ----------------------

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80,
        200,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        100,
        600,
        200
    )

with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Max Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# ---------------------- PREDICT ----------------------

if st.button("❤️ Predict Heart Disease"):

    raw = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    df = pd.DataFrame([raw])

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]

    st.write("")

    if prediction == 1:

        st.markdown(
            "<div class='bad'>⚠️ HIGH RISK OF HEART DISEASE</div>",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "<div class='ok'>✅ LOW RISK OF HEART DISEASE</div>",
            unsafe_allow_html=True
        )

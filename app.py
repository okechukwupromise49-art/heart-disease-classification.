
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# 2. Load the trained Random Forest model
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(
        "Unable to load the trained model. "
        "Check that heart_disease_model.pkl exists "
        "in your GitHub repository."
    )
    st.caption(f"Technical details: {error}")
    st.stop()

# --------------------------------------------------
# 3. App introduction
# --------------------------------------------------

st.title("❤️ Heart Disease Classification")

st.write(
    "This application uses a trained Random Forest "
    "machine-learning model to classify the presence "
    "or absence of heart disease based on 13 clinical features."
)

st.warning(
    "Educational demonstration only. "
    "This model is not a medical diagnostic tool "
    "and must not be used to make medical decisions."
)

with st.expander("About this project"):
    st.write(
        "This project compares Logistic Regression, "
        "Random Forest, and Multi-Layer Perceptron (MLP) "
        "models using the UCI Heart Disease dataset."
    )

    st.write("Selected model: Random Forest")
    st.write("Test accuracy: 85.00%")
    st.write("Test precision: 88.00%")
    st.write("Test recall: 78.57%")
    st.write("Test F1-score: 83.02%")

# --------------------------------------------------
# 4. Collect the 13 input features
# --------------------------------------------------

st.subheader("Enter the Clinical Features")

st.caption(
    "Use the feature definitions and coding from the "
    "original UCI dataset. These fields are intended "
    "for testing the model, not for self-diagnosis."
)

with st.form("heart_disease_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex (0 = female, 1 = male)",
            [0, 1]
        )

        cp = st.selectbox(
            "Chest pain type (cp)",
            [1, 2, 3, 4]
        )

        trestbps = st.number_input(
            "Resting blood pressure (trestbps)",
            min_value=50,
            max_value=300,
            value=120
        )

        chol = st.number_input(
            "Serum cholesterol (chol)",
            min_value=50,
            max_value=700,
            value=200
        )

        fbs = st.selectbox(
            "Fasting blood sugar above 120 mg/dl (fbs)",
            [0, 1]
        )

        restecg = st.selectbox(
            "Resting ECG result (restecg)",
            [0, 1, 2]
        )

    with col2:
        thalach = st.number_input(
            "Maximum heart rate achieved (thalach)",
            min_value=50,
            max_value=250,
            value=150
        )

        exang = st.selectbox(
            "Exercise-induced angina (exang)",
            [0, 1]
        )

        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "ST segment slope (slope)",
            [1, 2, 3]
        )

        ca = st.selectbox(
            "Number of major vessels (ca)",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thal category (thal)",
            [3, 6, 7]
        )

    submitted = st.form_submit_button(
        "Classify Heart Disease",
        use_container_width=True
    )

# --------------------------------------------------
# 5. Make a prediction
# --------------------------------------------------

if submitted:

    # Feature names and order must match model training.
    input_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    )

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Classification Result")

        if prediction == 1:
            st.error(
                "The model classified this input as "
                "indicating the presence of heart disease."
            )
        else:
            st.success(
                "The model classified this input as "
                "indicating the absence of heart disease."
            )

        st.caption(
            "This result is an experimental model output, "
            "not a medical diagnosis. Consult a qualified "
            "healthcare professional about health concerns."
        )

    except Exception as error:
        st.error("Prediction failed.")
        st.caption(f"Technical details: {error}")

# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Project: Heart Disease Classification | "
    "Dataset: UCI Machine Learning Repository | "
    "Model: Random Forest"
)


import streamlit as st

st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Classification")
st.write(
    "A machine learning project that classifies "
    "the presence or absence of heart disease "
    "using 13 clinical features."
)

st.warning(
    "Educational demonstration only. "
    "This app is not a medical diagnostic tool."
)

st.subheader("About the Project")

st.write("""
This project compares three machine learning models:
- Logistic Regression
- Random Forest
- Multi-Layer Perceptron (MLP)

The models are evaluated using accuracy,
precision, recall, F1-score, and confusion matrices.
""")

st.info(
    "Next: connect the trained model and add "
    "the input fields for all 13 features."
)

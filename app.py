import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="HeartGuard AI | Heart Disease Classification",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. Custom CSS – Modern Medical Theme
# --------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem;
        padding: 0.4rem 0;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.07), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.6rem;
        color: #0f172a;
    }
    .metric-card p {
        margin: 0.25rem 0 0 0;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Result boxes */
    .result-positive {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 6px solid #ef4444;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 6px solid #22c55e;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Form labels */
    label {
        font-weight: 500 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #b91c1c 0%, #991b1b 100%);
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.35);
        transform: translateY(-1px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 1.5rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model. Please ensure `heart_disease_model.pkl` is in the same folder.")
    st.caption(f"Technical details: {error}")
    st.stop()

# --------------------------------------------------
# 4. Sidebar Navigation
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ❤️ HeartGuard AI")
    st.markdown("#### Heart Disease Classifier")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔬 Predict", "📊 Model Info", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**Selected Model**")
    st.markdown("Random Forest Classifier")
    st.markdown("")
    st.caption("Educational demo only · Not for medical use")

# --------------------------------------------------
# 5. HOME PAGE
# --------------------------------------------------
if page == "🏠 Home":
    st.markdown("# ❤️ HeartGuard AI")
    st.markdown("### Intelligent Heart Disease Risk Classification")
    
    st.markdown("""
    Welcome to **HeartGuard AI** — a modern web application that uses a trained 
    **Random Forest** model to classify the presence or absence of heart disease 
    based on 13 clinical features from the UCI Heart Disease dataset.
    """)
    
    st.warning("⚠️ **Important Disclaimer**  \nThis tool is an **educational demonstration only**. "
               "It is **not** a medical diagnostic device and must never be used to make real medical decisions. "
               "Always consult a qualified healthcare professional.")
    
    st.markdown("### Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>85.0%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>88.0%</h3><p>Precision</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>78.6%</h3><p>Recall</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>94.1%</h3><p>ROC-AUC</p></div>', unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("### How to use")
    st.markdown("""
    1. Go to the **🔬 Predict** page in the sidebar  
    2. Enter the 13 clinical features  
    3. Click **Classify Heart Disease**  
    4. View the model’s prediction (educational only)
    """)

# --------------------------------------------------
# 6. PREDICT PAGE
# --------------------------------------------------
elif page == "🔬 Predict":
    st.markdown("# 🔬 Heart Disease Classification")
    st.markdown("Enter the clinical features below. All values follow the original UCI Heart Disease dataset coding.")
    
    with st.form("heart_disease_form"):
        st.markdown('<div class="section-header">👤 Patient Demographics & Vitals</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=54, help="Patient age in years")
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=300, value=130)
        
        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=700, value=250)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                               format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
            thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
        
        with col3:
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            exang = st.selectbox("Exercise-Induced Angina", options=[0, 1],
                                 format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
            restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                                   format_func=lambda x: {0: "Normal (0)", 1: "ST-T abnormality (1)", 2: "Left ventricular hypertrophy (2)"}[x])
        
        st.markdown("---")
        st.markdown('<div class="section-header">🩺 Clinical Features</div>', unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4],
                              format_func=lambda x: {
                                  1: "Typical angina (1)",
                                  2: "Atypical angina (2)",
                                  3: "Non-anginal pain (3)",
                                  4: "Asymptomatic (4)"
                              }[x])
        
        with col5:
            slope = st.selectbox("ST Segment Slope", options=[1, 2, 3],
                                 format_func=lambda x: {
                                     1: "Upsloping (1)",
                                     2: "Flat (2)",
                                     3: "Downsloping (3)"
                                 }[x])
            ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3])
        
        with col6:
            thal = st.selectbox("Thalassemia (thal)", options=[3, 6, 7],
                                format_func=lambda x: {
                                    3: "Normal (3)",
                                    6: "Fixed defect (6)",
                                    7: "Reversible defect (7)"
                                }[x])
        
        st.markdown("")
        submitted = st.form_submit_button("🩺 Classify Heart Disease", use_container_width=True)
    
    # Prediction logic
    if submitted:
        input_data = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        )
        
        try:
            prediction = model.predict(input_data)[0]
            
            st.markdown("### Classification Result")
            
            if prediction == 1:
                st.markdown("""
                <div class="result-positive">
                    <h3 style="color:#b91c1c; margin-top:0;">⚠️ Presence of Heart Disease Detected</h3>
                    <p>The Random Forest model classified this input as <strong>indicating the presence of heart disease</strong>.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative">
                    <h3 style="color:#15803d; margin-top:0;">✅ Absence of Heart Disease</h3>
                    <p>The Random Forest model classified this input as <strong>indicating the absence of heart disease</strong>.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.info("This result is an **experimental model output only**. It is not a medical diagnosis. "
                    "Please consult a qualified healthcare professional for any health concerns.")
            
            # Optional: show the input values
            with st.expander("View submitted features"):
                st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
                
        except Exception as error:
            st.error("Prediction failed.")
            st.caption(f"Technical details: {error}")

# --------------------------------------------------
# 7. MODEL INFO PAGE
# --------------------------------------------------
elif page == "📊 Model Info":
    st.markdown("# 📊 Model Performance")
    st.markdown("The following metrics were obtained on the held-out test set.")
    
    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "ROC-AUC"],
        "Score": ["85.00%", "88.00%", "78.57%", "83.02%", "80.29%", "94.08%"]
    }
    st.table(pd.DataFrame(metrics))
    
    st.markdown("### Model Comparison Summary")
    st.markdown("""
    Three models were trained and evaluated on the UCI Heart Disease dataset:
    
    | Model                  | Accuracy | Notes                          |
    |------------------------|----------|--------------------------------|
    | Logistic Regression    | —        | Baseline linear model          |
    | **Random Forest**      | **85%**  | **Selected final model**       |
    | Multi-Layer Perceptron | —        | Neural network baseline        |
    
    Random Forest was chosen for its strong balance of accuracy, precision, and robustness.
    """)
    
    st.markdown("### Feature List (13 clinical attributes)")
    features = [
        ("age", "Age in years"),
        ("sex", "Sex (0 = female, 1 = male)"),
        ("cp", "Chest pain type (1–4)"),
        ("trestbps", "Resting blood pressure (mm Hg)"),
        ("chol", "Serum cholesterol (mg/dl)"),
        ("fbs", "Fasting blood sugar > 120 mg/dl (0/1)"),
        ("restecg", "Resting electrocardiographic results (0–2)"),
        ("thalach", "Maximum heart rate achieved"),
        ("exang", "Exercise induced angina (0/1)"),
        ("oldpeak", "ST depression induced by exercise"),
        ("slope", "Slope of the peak exercise ST segment (1–3)"),
        ("ca", "Number of major vessels colored by fluoroscopy (0–3)"),
        ("thal", "Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)"),
    ]
    st.table(pd.DataFrame(features, columns=["Feature", "Description"]))

# --------------------------------------------------
# 8. ABOUT PAGE
# --------------------------------------------------
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")
    
    st.markdown("""
    ### Project Overview
    This application demonstrates a complete machine-learning pipeline for binary 
    classification of heart disease using the classic **UCI Heart Disease** dataset.
    
    ### Models Compared
    - Logistic Regression
    - Random Forest Classifier (**selected**)
    - Multi-Layer Perceptron (MLP)
    
    ### Final Model
    - **Algorithm**: Random Forest  
    - **Test Accuracy**: 85.00%  
    - **Test ROC-AUC**: 94.08%
    
    ### Important Notice
    This tool is built **strictly for educational and demonstration purposes**.  
    It must **never** be used as a substitute for professional medical advice, diagnosis, or treatment.
    """)
    
    st.markdown("---")
    st.markdown("**Dataset source**: UCI Machine Learning Repository – Heart Disease dataset")

# --------------------------------------------------
# 9. Footer (all pages)
# --------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="footer">'
    'HeartGuard AI · Educational Demonstration · Dataset: UCI Heart Disease · Model: Random Forest'
    '</div>',
    unsafe_allow_html=True
)

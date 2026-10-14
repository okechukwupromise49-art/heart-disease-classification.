import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="HeartGuard AI | Clinical Decision Support",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. Dark Modern Professional CSS
# --------------------------------------------------
st.markdown("""
<style>
    /* ===== GLOBAL ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #0a0a0f;
        color: #e2e8f0;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f17 0%, #15151f 100%);
        border-right: 1px solid #1e1e2e;
    }
    
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 0.95rem;
        font-weight: 500;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 4px;
        transition: all 0.2s ease;
    }
    
    /* ===== CARDS ===== */
    .glass-card {
        background: rgba(20, 20, 30, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-box {
        background: linear-gradient(145deg, #14141f, #1a1a28);
        border: 1px solid #2a2a3a;
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-3px);
        border-color: #ef4444;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f87171, #ef4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.3rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* ===== RESULT CARDS ===== */
    .result-danger {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.25), rgba(153, 27, 27, 0.15));
        border: 1px solid #7f1d1d;
        border-left: 5px solid #ef4444;
        border-radius: 14px;
        padding: 1.8rem;
        margin: 1.2rem 0;
    }
    
    .result-safe {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.25), rgba(4, 120, 87, 0.15));
        border: 1px solid #065f46;
        border-left: 5px solid #10b981;
        border-radius: 14px;
        padding: 1.8rem;
        margin: 1.2rem 0;
    }
    
    /* ===== SECTION TITLE ===== */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    .section-title::before {
        content: "";
        width: 4px;
        height: 18px;
        background: #ef4444;
        border-radius: 2px;
    }
    
    /* ===== BUTTON ===== */
    .stButton > button {
        background: linear-gradient(90deg, #dc2626, #b91c1c) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.25);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #ef4444, #dc2626) !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4) !important;
        transform: translateY(-2px);
    }
    
    /* ===== INPUTS ===== */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #14141f !important;
        border: 1px solid #2a2a3a !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* ===== FOOTER ===== */
    .custom-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        letter-spacing: 0.5px;
    }
    
    /* ===== HEADERS ===== */
    h1, h2, h3 {
        color: #f8fafc !important;
    }
    
    .stMarkdown p {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. Load Model
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Failed to load model. Ensure `heart_disease_model.pkl` is present.")
    st.caption(str(e))
    st.stop()

# --------------------------------------------------
# 4. Sidebar Navigation
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
            <div style="font-size: 2.2rem;">❤️</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.5px;">
                HeartGuard AI
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">
                CLINICAL DECISION SUPPORT
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Prediction", "Model Performance", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.6;">
            <strong style="color:#94a3b8;">Active Model</strong><br>
            Random Forest Classifier<br><br>
            <strong style="color:#94a3b8;">Test Accuracy</strong><br>
            85.00%
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Educational use only • Not a medical device")

# --------------------------------------------------
# 5. DASHBOARD
# --------------------------------------------------
if page == "Dashboard":
    st.markdown("# HeartGuard AI")
    st.markdown("##### Advanced clinical decision support system powered by Random Forest")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-box">
                <div class="metric-value">85.0%</div>
                <div class="metric-label">Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-box">
                <div class="metric-value">88.0%</div>
                <div class="metric-label">Precision</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-box">
                <div class="metric-value">78.6%</div>
                <div class="metric-label">Recall</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-box">
                <div class="metric-value">94.1%</div>
                <div class="metric-label">ROC-AUC</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Info cards
    left, right = st.columns(2)
    
    with left:
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top:0; color:#f8fafc;">About the System</h3>
                <p style="color:#94a3b8; line-height:1.7;">
                    HeartGuard AI uses a carefully tuned Random Forest model trained on the 
                    classic UCI Heart Disease dataset. It analyzes 13 clinical parameters 
                    to provide a binary classification of heart disease presence.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with right:
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top:0; color:#f8fafc;">Important Notice</h3>
                <p style="color:#94a3b8; line-height:1.7;">
                    This application is designed strictly for educational and research purposes. 
                    It is <strong style="color:#f87171;">not a medical diagnostic tool</strong> and must never 
                    be used to make clinical decisions. Always consult qualified medical professionals.
                </p>
            </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# 6. PREDICTION PAGE
# --------------------------------------------------
elif page == "Prediction":
    st.markdown("# Clinical Prediction")
    st.markdown("Enter patient clinical parameters below. Values follow UCI Heart Disease coding.")
    
    with st.form("prediction_form"):
        
        st.markdown('<div class="section-title">Patient Demographics & Vitals</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("Age (years)", 1, 120, 54)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            trestbps = st.number_input("Resting BP (mm Hg)", 50, 300, 130)
        
        with c2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", 50, 700, 250)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes")
            thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
        
        with c3:
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
            exang = st.selectbox("Exercise Induced Angina", [0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG", [0, 1, 2],
                                   format_func=lambda x: {
                                       0: "Normal",
                                       1: "ST-T Abnormality",
                                       2: "LV Hypertrophy"
                                   }[x])
        
        st.markdown('<div class="section-title">Advanced Clinical Features</div>', unsafe_allow_html=True)
        
        c4, c5, c6 = st.columns(3)
        
        with c4:
            cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                              format_func=lambda x: {
                                  1: "Typical Angina",
                                  2: "Atypical Angina",
                                  3: "Non-anginal Pain",
                                  4: "Asymptomatic"
                              }[x])
        
        with c5:
            slope = st.selectbox("ST Segment Slope", [1, 2, 3],
                                 format_func=lambda x: {
                                     1: "Upsloping",
                                     2: "Flat",
                                     3: "Downsloping"
                                 }[x])
            ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
        
        with c6:
            thal = st.selectbox("Thalassemia", [3, 6, 7],
                                format_func=lambda x: {
                                    3: "Normal",
                                    6: "Fixed Defect",
                                    7: "Reversible Defect"
                                }[x])
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Classification", use_container_width=True)
    
    if submitted:
        input_df = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]], columns=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ])
        
        try:
            pred = model.predict(input_df)[0]
            
            st.markdown("### Classification Result")
            
            if pred == 1:
                st.markdown("""
                    <div class="result-danger">
                        <h2 style="margin:0 0 0.5rem 0; color:#f87171;">⚠ Presence of Heart Disease</h2>
                        <p style="margin:0; color:#fca5a5; font-size:1.05rem;">
                            The model indicates a high likelihood of heart disease based on the provided clinical features.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-safe">
                        <h2 style="margin:0 0 0.5rem 0; color:#34d399;">✓ Absence of Heart Disease</h2>
                        <p style="margin:0; color:#6ee7b7; font-size:1.05rem;">
                            The model indicates a low likelihood of heart disease based on the provided clinical features.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.info("This is an experimental model output for educational purposes only. It is not a medical diagnosis.")
            
            with st.expander("View Input Features"):
                st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
                
        except Exception as e:
            st.error("Prediction failed.")
            st.caption(str(e))

# --------------------------------------------------
# 7. MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    st.markdown("# Model Performance")
    st.markdown("Detailed evaluation metrics on the held-out test set.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "ROC-AUC"],
        "Value": ["85.00%", "88.00%", "78.57%", "83.02%", "80.29%", "94.08%"]
    }
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### Feature Descriptions")
    
    features = [
        ("age", "Age in years"),
        ("sex", "Biological sex (0 = female, 1 = male)"),
        ("cp", "Chest pain type (1–4)"),
        ("trestbps", "Resting blood pressure (mm Hg)"),
        ("chol", "Serum cholesterol (mg/dl)"),
        ("fbs", "Fasting blood sugar > 120 mg/dl"),
        ("restecg", "Resting electrocardiographic results"),
        ("thalach", "Maximum heart rate achieved"),
        ("exang", "Exercise-induced angina"),
        ("oldpeak", "ST depression induced by exercise"),
        ("slope", "Slope of the peak exercise ST segment"),
        ("ca", "Number of major vessels (0–3)"),
        ("thal", "Thalassemia category"),
    ]
    
    st.dataframe(pd.DataFrame(features, columns=["Feature", "Description"]), 
                 use_container_width=True, hide_index=True)

# --------------------------------------------------
# 8. ABOUT
# --------------------------------------------------
elif page == "About":
    st.markdown("# About HeartGuard AI")
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0;">Project Overview</h3>
        <p style="color:#94a3b8; line-height:1.7;">
            HeartGuard AI is an educational demonstration of a complete machine learning 
            pipeline for binary classification of heart disease using the UCI Heart Disease dataset.
        </p>
        
        <h3>Models Evaluated</h3>
        <ul style="color:#94a3b8; line-height:1.8;">
            <li>Logistic Regression</li>
            <li><strong style="color:#f87171;">Random Forest Classifier</strong> (Selected)</li>
            <li>Multi-Layer Perceptron (Neural Network)</li>
        </ul>
        
        <h3>Final Model Performance</h3>
        <p style="color:#94a3b8;">
            Test Accuracy: <strong style="color:#f8fafc;">85.00%</strong><br>
            Test ROC-AUC: <strong style="color:#f8fafc;">94.08%</strong>
        </p>
        
        <h3>Disclaimer</h3>
        <p style="color:#f87171;">
            This system is intended solely for educational and research purposes. 
            It is not a certified medical device and must not be used for real clinical diagnosis or treatment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# 9. Footer
# --------------------------------------------------
st.markdown("""
    <div class="custom-footer">
        HEARTGUARD AI  •  EDUCATIONAL DEMONSTRATION  •  UCI HEART DISEASE DATASET  •  RANDOM FOREST
    </div>
""", unsafe_allow_html=True)

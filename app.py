import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==================================================
# 1. Page configuration
# ==================================================

st.set_page_config(
    page_title="CardioScreen | Heart Disease Classification",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# 2. Design system (CSS)
# ==================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #0B2A4A;
        --teal: #0E8F8A;
        --teal-dark: #0A6F6B;
        --teal-soft: #E3F4F3;
        --bg: #F3F7FA;
        --card: #FFFFFF;
        --line: #DCE5EC;
        --text: #1C2B39;
        --muted: #5F7385;
        --danger: #C0392B;
        --danger-soft: #FCEBE9;
        --ok: #1B8A5A;
        --ok-soft: #E4F5EC;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
        color: var(--text);
    }
    .stApp { background: var(--bg); }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    /* ---------- Top navigation bar ---------- */
    .topbar {
        background: var(--navy);
        border-radius: 14px 14px 0 0;
        padding: 16px 26px 14px 26px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
        width: 38px; height: 38px; border-radius: 10px;
        background: var(--teal);
        display: flex; align-items: center; justify-content: center;
        font-size: 20px;
    }
    .brand-name { color: #fff; font-weight: 800; font-size: 1.15rem; letter-spacing: .2px; }
    .brand-sub { color: #9FB6CC; font-size: .75rem; font-weight: 500; }
    .topbar-badge {
        color: #CFE3F3; font-size: .78rem; font-weight: 600;
        border: 1px solid rgba(255,255,255,.22);
        padding: 5px 12px; border-radius: 999px;
    }

    /* Radio group styled as nav tabs */
    div[data-testid="stRadio"] {
        background: var(--navy);
        border-radius: 0 0 14px 14px;
        padding: 0 20px 12px 20px;
        margin-bottom: 1.6rem;
    }
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] div[role="radiogroup"] { gap: 6px; }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: transparent;
        padding: 8px 18px;
        border-radius: 8px;
        cursor: pointer;
        transition: background .15s ease;
        margin: 0;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none; }
    div[data-testid="stRadio"] label[data-baseweb="radio"] p {
        color: #B9CCDD; font-weight: 600; font-size: .92rem; margin: 0;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover { background: rgba(255,255,255,.08); }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) { background: var(--teal); }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p { color: #fff; }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(120deg, #0B2A4A 0%, #0E5F73 60%, #0E8F8A 100%);
        border-radius: 16px;
        padding: 46px 44px;
        color: #fff;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.3rem; font-weight: 800; line-height: 1.2;
        margin: 0 0 12px 0; color: #fff; padding: 0;
    }
    .hero p { color: #D5E8F2; font-size: 1.02rem; max-width: 620px; margin: 0; line-height: 1.65; }

    /* ---------- Cards ---------- */
    .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 22px 24px;
        height: 100%;
    }
    .card h4 { margin: 0 0 6px 0; font-size: 1rem; font-weight: 700; color: var(--navy); }
    .card p { margin: 0; color: var(--muted); font-size: .9rem; line-height: 1.6; }

    .stat {
        background: var(--card);
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 12px;
        padding: 16px 18px;
    }
    .stat .label { color: var(--muted); font-size: .8rem; font-weight: 600; }
    .stat .value { color: var(--navy); font-size: 1.7rem; font-weight: 800; margin-top: 2px; }

    .section-title {
        font-size: 1.25rem; font-weight: 800; color: var(--navy);
        margin: 0 0 4px 0;
    }
    .section-sub { color: var(--muted); font-size: .92rem; margin-bottom: 1rem; }

    .group-head {
        font-weight: 700; color: var(--teal-dark); font-size: .95rem;
        padding-bottom: 6px; margin: 6px 0 10px 0;
        border-bottom: 2px solid var(--teal-soft);
    }

    /* ---------- Result ---------- */
    .result {
        border-radius: 14px; padding: 24px 26px; margin-top: 1rem;
        border: 1px solid; 
    }
    .result.pos { background: var(--danger-soft); border-color: #F0B9B3; }
    .result.neg { background: var(--ok-soft); border-color: #B4E0C8; }
    .result h3 { margin: 0 0 6px 0; font-size: 1.2rem; font-weight: 800; }
    .result.pos h3 { color: var(--danger); }
    .result.neg h3 { color: var(--ok); }
    .result p { margin: 0; color: var(--text); font-size: .95rem; line-height: 1.6; }

    .notice {
        background: #FFF7E6; border: 1px solid #F3DDA6; color: #6B4E0A;
        border-radius: 12px; padding: 14px 18px; font-size: .88rem; line-height: 1.55;
    }

    /* ---------- Inputs and buttons ---------- */
    div[data-testid="stForm"] {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 24px 26px;
    }
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: var(--teal);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.7rem 1.2rem;
        transition: background .15s ease;
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: var(--teal-dark); color: #fff;
    }
    .stButton > button:focus-visible,
    div[data-testid="stFormSubmitButton"] > button:focus-visible {
        outline: 3px solid #9ADAD6; outline-offset: 2px;
    }

    .footer {
        text-align: center; color: var(--muted); font-size: .8rem;
        margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line);
    }

    @media (max-width: 700px) {
        .hero { padding: 30px 22px; }
        .hero h1 { font-size: 1.7rem; }
        .topbar { flex-direction: column; align-items: flex-start; gap: 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 3. Load the trained Random Forest model
# ==================================================

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

# ==================================================
# 4. Reference data
# ==================================================

METRICS = {
    "Accuracy": 85.00,
    "Precision": 88.00,
    "Recall": 78.57,
    "F1-score": 83.02,
    "F2-score": 80.29,
    "ROC-AUC": 94.08,
}

LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-anginal pain",
        4: "Asymptomatic",
    },
    "fbs": {0: "No (120 mg/dl or below)", 1: "Yes (above 120 mg/dl)"},
    "restecg": {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Left ventricular hypertrophy",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

PAGES = ["Home", "Classify", "Model Performance", "About"]

# ==================================================
# 5. Top navigation bar
# ==================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark">🫀</div>
            <div>
                <div class="brand-name">CardioScreen</div>
                <div class="brand-sub">Heart disease classification</div>
            </div>
        </div>
        <div class="topbar-badge">Educational demo</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    PAGES,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_page",
)

NOTICE = (
    "<div class='notice'><b>Educational demonstration only.</b> "
    "This model is not a medical diagnostic tool and must not be used "
    "to make medical decisions.</div>"
)

# ==================================================
# 6. Pages
# ==================================================


def page_home():
    st.markdown(
        """
        <div class="hero">
            <h1>Machine-learning support for heart disease classification</h1>
            <p>Enter 13 clinical features and a trained Random Forest model
            classifies the input as showing the presence or absence of heart
            disease. Built on the UCI Heart Disease dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, value) in zip(
        (c1, c2, c3, c4),
        [
            ("Test accuracy", f"{METRICS['Accuracy']:.2f}%"),
            ("ROC-AUC", f"{METRICS['ROC-AUC']:.2f}%"),
            ("Clinical features", "13"),
            ("Selected model", "Random Forest"),
        ],
    ):
        with col:
            st.markdown(
                f"<div class='stat'><div class='label'>{label}</div>"
                f"<div class='value' style='font-size:{'1.7rem' if len(value) < 9 else '1.2rem'}'>{value}</div></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            "<div class='card'><h4>Enter clinical features</h4>"
            "<p>Fill in age, blood pressure, cholesterol, ECG results and "
            "the other inputs in the Classify tab.</p></div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            "<div class='card'><h4>Get a classification</h4>"
            "<p>The model returns presence or absence of heart disease, "
            "with a confidence estimate when available.</p></div>",
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            "<div class='card'><h4>Review model quality</h4>"
            "<p>See accuracy, precision, recall, F-scores and ROC-AUC "
            "in the Model Performance tab.</p></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(NOTICE, unsafe_allow_html=True)


def page_classify():
    st.markdown("<div class='section-title'>Patient clinical features</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Use the feature definitions and coding from the "
        "original UCI dataset. These fields are for testing the model, "
        "not for self-diagnosis.</div>",
        unsafe_allow_html=True,
    )

    with st.form("heart_disease_form"):
        st.markdown("<div class='group-head'>Demographics and vitals</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        with c2:
            sex = st.selectbox("Sex", list(LABELS["sex"]), format_func=LABELS["sex"].get)
        with c3:
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)", min_value=50, max_value=300, value=120
            )
        with c4:
            chol = st.number_input(
                "Serum cholesterol (mg/dl)", min_value=50, max_value=700, value=200
            )

        st.markdown("<div class='group-head'>Symptoms and blood tests</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            cp = st.selectbox("Chest pain type", list(LABELS["cp"]), format_func=LABELS["cp"].get)
        with c2:
            fbs = st.selectbox(
                "Fasting blood sugar above 120 mg/dl",
                list(LABELS["fbs"]),
                format_func=LABELS["fbs"].get,
            )
        with c3:
            exang = st.selectbox(
                "Exercise-induced angina", list(LABELS["exang"]), format_func=LABELS["exang"].get
            )

        st.markdown("<div class='group-head'>ECG and stress test</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            restecg = st.selectbox(
                "Resting ECG result", list(LABELS["restecg"]), format_func=LABELS["restecg"].get
            )
        with c2:
            thalach = st.number_input(
                "Maximum heart rate achieved (bpm)", min_value=50, max_value=250, value=150
            )
        with c3:
            oldpeak = st.number_input(
                "ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1
            )

        st.markdown("<div class='group-head'>Imaging</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            slope = st.selectbox(
                "ST segment slope", list(LABELS["slope"]), format_func=LABELS["slope"].get
            )
        with c2:
            ca = st.selectbox(
                "Major vessels coloured by fluoroscopy",
                list(LABELS["ca"]),
                format_func=LABELS["ca"].get,
            )
        with c3:
            thal = st.selectbox("Thalassemia (thal)", list(LABELS["thal"]), format_func=LABELS["thal"].get)

        submitted = st.form_submit_button("Classify heart disease", use_container_width=True)

    if submitted:
        # Feature names and order must match model training.
        input_data = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]],
            columns=FEATURE_ORDER,
        )

        try:
            prediction = model.predict(input_data)[0]

            probability = None
            if hasattr(model, "predict_proba"):
                try:
                    probability = float(model.predict_proba(input_data)[0][1])
                except Exception:
                    probability = None

            st.markdown("<div class='section-title' style='margin-top:1.4rem'>Classification result</div>", unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(
                    "<div class='result pos'><h3>Presence of heart disease indicated</h3>"
                    "<p>The model classified this input as indicating the presence "
                    "of heart disease.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result neg'><h3>Absence of heart disease indicated</h3>"
                    "<p>The model classified this input as indicating the absence "
                    "of heart disease.</p></div>",
                    unsafe_allow_html=True,
                )

            if probability is not None:
                st.write("")
                st.markdown(
                    f"**Model estimated probability of heart disease: {probability * 100:.1f}%**"
                )
                st.progress(min(max(probability, 0.0), 1.0))

            with st.expander("View submitted values"):
                st.dataframe(input_data, use_container_width=True, hide_index=True)

            st.caption(
                "This result is an experimental model output, not a medical "
                "diagnosis. Consult a qualified healthcare professional about "
                "health concerns."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.caption(f"Technical details: {error}")


def page_performance():
    st.markdown("<div class='section-title'>Model performance</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Random Forest results on the held-out test set.</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for i, (name, value) in enumerate(METRICS.items()):
        with cols[i % 3]:
            st.markdown(
                f"<div class='stat'><div class='label'>{name}</div>"
                f"<div class='value'>{value:.2f}%</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")

    st.markdown("<div class='group-head'>Metric comparison</div>", unsafe_allow_html=True)
    chart_df = pd.DataFrame({"Score (%)": METRICS})
    st.bar_chart(chart_df, color="#0E8F8A", height=320)

    st.markdown(
        "<div class='card'><h4>How to read these scores</h4>"
        "<p>Recall shows how many true heart disease cases the model finds, while "
        "precision shows how many of its positive calls are correct. The F2-score "
        "weights recall more heavily, which matters when missing a case is costly. "
        "ROC-AUC summarises how well the model separates the two classes across "
        "all thresholds.</p></div>",
        unsafe_allow_html=True,
    )


def page_about():
    st.markdown("<div class='section-title'>About this project</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Dataset, models and feature reference.</div>",
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)
    with a:
        st.markdown(
            "<div class='card'><h4>Project</h4>"
            "<p>This project compares Logistic Regression, Random Forest, and "
            "Multi-Layer Perceptron (MLP) models using the UCI Heart Disease "
            "dataset. The Random Forest was selected as the final model.</p></div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            "<div class='card'><h4>Intended use</h4>"
            "<p>The app demonstrates how a trained model can be deployed for "
            "classification. It is for learning and testing only.</p></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("<div class='group-head'>Feature reference</div>", unsafe_allow_html=True)
    reference = pd.DataFrame(
        {
            "Feature": FEATURE_ORDER,
            "Description": [
                "Age in years",
                "Sex (0 = female, 1 = male)",
                "Chest pain type (1 to 4)",
                "Resting blood pressure",
                "Serum cholesterol",
                "Fasting blood sugar above 120 mg/dl",
                "Resting ECG result (0 to 2)",
                "Maximum heart rate achieved",
                "Exercise-induced angina",
                "ST depression induced by exercise",
                "Slope of the peak exercise ST segment (1 to 3)",
                "Number of major vessels (0 to 3)",
                "Thal category (3, 6 or 7)",
            ],
        }
    )
    st.dataframe(reference, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown(NOTICE, unsafe_allow_html=True)


if page == "Home":
    page_home()
elif page == "Classify":
    page_classify()
elif page == "Model Performance":
    page_performance()
else:
    page_about()

# ==================================================
# 7. Footer
# ==================================================

st.markdown(
    "<div class='footer'>Project: Heart Disease Classification &nbsp;|&nbsp; "
    "Dataset: UCI Machine Learning Repository &nbsp;|&nbsp; Model: Random Forest</div>",
    unsafe_allow_html=True,
)


import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CardioSense AI | Heart Disease Classification",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 2. CUSTOM CSS — MODERN HEALTHCARE DESIGN
# =========================================================

st.markdown(
    """
    <style>
    /* Global layout */
    .stApp {
        background-color: #F5F8FC;
        color: #172B4D;
    }

    [data-testid="stHeader"] {
        background-color: rgba(245, 248, 252, 0.95);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.3rem;
        padding-bottom: 3rem;
    }

    /* Typography */
    h1, h2, h3 {
        color: #142D4E !important;
        letter-spacing: -0.5px;
    }

    p, label, span {
        line-height: 1.6;
    }

    /* Top navigation brand */
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }

    .brand-icon {
        background: #DDF7F0;
        color: #087F70;
        border-radius: 14px;
        padding: 10px 13px;
        font-size: 27px;
    }

    .brand-name {
        font-size: 24px;
        font-weight: 800;
        color: #142D4E;
        line-height: 1.2;
    }

    .brand-subtitle {
        color: #718198;
        font-size: 12px;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(115deg, #12365A, #087F8C);
        color: white;
        padding: 36px;
        border-radius: 22px;
        margin: 18px 0 24px 0;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(28px, 4vw, 42px);
        margin-bottom: 12px;
    }

    .hero p {
        color: #E2F4F7;
        font-size: 16px;
        max-width: 760px;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.14);
        color: #E8FFFA;
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 12px;
        margin-bottom: 15px;
        border: 1px solid rgba(255,255,255,0.22);
    }

    /* Cards */
    .panel {
        background: #FFFFFF;
        border: 1px solid #E5ECF4;
        border-radius: 18px;
        padding: 23px;
        margin-bottom: 18px;
        box-shadow: 0 5px 20px rgba(25, 55, 90, 0.035);
    }

    .panel-title {
        font-size: 19px;
        font-weight: 750;
        color: #142D4E;
        margin-bottom: 5px;
    }

    .muted {
        color: #718198;
        font-size: 13px;
    }

    /* Metrics */
    .metric-card {
        background: white;
        border: 1px solid #E5ECF4;
        border-radius: 16px;
        padding: 20px;
        min-height: 115px;
        box-shadow: 0 4px 16px rgba(25,55,90,0.025);
    }

    .metric-label {
        color: #718198;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #142D4E;
        font-size: 28px;
        font-weight: 800;
    }

    .metric-note {
        color: #087F70;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Notice */
    .notice {
        background: #FFF8E8;
        border: 1px solid #F1D9A8;
        border-radius: 12px;
        padding: 15px 18px;
        color: #77531A;
        font-size: 13px;
        margin: 16px 0 22px 0;
    }

    /* Result panels */
    .result-positive {
        background: #FFF1F0;
        border: 1px solid #F3C5C1;
        border-radius: 16px;
        padding: 24px;
    }

    .result-negative {
        background: #EAF8F2;
        border: 1px solid #B8E5D1;
        border-radius: 16px;
        padding: 24px;
    }

    .result-title {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    /* Form fields */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 9px;
        border-color: #D9E3EF;
        background-color: white;
    }

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #E5ECF4;
        border-radius: 18px;
        padding: 24px;
    }

    /* Buttons */
    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #087F70;
        color: white;
        border: 1px solid #087F70;
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #06695E;
        color: white;
        border-color: #06695E;
    }

    /* Navigation radio */
    div[data-testid="stRadio"] > div {
        gap: 8px;
    }

    div[data-testid="stRadio"] label {
        background: white;
        border: 1px solid #E2EAF3;
        border-radius: 9px;
        padding: 8px 15px;
        margin-bottom: 5px;
    }

    /* Footer */
    .footer {
        border-top: 1px solid #E2EAF3;
        margin-top: 40px;
        padding-top: 20px;
        color: #718198;
        font-size: 12px;
        text-align: center;
    }

    /* Small screens */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }

        .hero {
            padding: 24px;
        }

        .panel {
            padding: 17px;
        }

        .metric-card {
            padding: 14px;
        }

        .metric-value {
            font-size: 23px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 3. MODEL LOADING
# =========================================================

MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception:
    st.error(
        "The trained model could not be loaded. "
        "Ensure heart_disease_model.pkl is in the same "
        "directory as app.py and that the file is valid."
    )
    st.info(
        "Check your repository structure and the model "
        "file's name. Review deployment logs for details."
    )
    st.stop()

# =========================================================
# 4. REUSABLE UI HELPERS
# =========================================================

def metric_card(label, value, note="Model evaluation"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_heading(title, description):
    st.markdown(
        f"""
        <div style="margin: 10px 0 20px 0;">
            <div style="font-size:27px;font-weight:800;color:#142D4E;">
                {title}
            </div>
            <div style="color:#718198;font-size:14px;margin-top:5px;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_footer():
    st.markdown(
        """
        <div class="footer">
            <strong>CardioSense AI</strong> · Heart Disease Classification
            <br>
            Research demonstration · UCI Machine Learning Repository
            <br>
            This tool is not intended for medical diagnosis or treatment.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 5. NAVIGATION BAR
# =========================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">♥</div>
            <div>
                <div class="brand-name">CardioSense AI</div>
                <div class="brand-subtitle">
                    MACHINE LEARNING RESEARCH DASHBOARD
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right;padding-top:12px;">
            <span style="
                background:#DDF7F0;
                color:#087F70;
                padding:8px 12px;
                border-radius:20px;
                font-size:12px;
                font-weight:700;">
                ● Research Demo
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

page = st.radio(
    "Main navigation",
    [
        "Overview",
        "Classifier",
        "Model Performance",
        "About"
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation"
)

st.divider()

# =========================================================
# 6. SHARED INTRODUCTION
# =========================================================

st.markdown(
    """
    <div class="notice">
        <strong>Educational use only.</strong>
        This application produces experimental machine-learning
        classifications. It is not a validated clinical diagnostic
        tool and must not be used to make medical decisions.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 7. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">HEALTHCARE × ARTIFICIAL INTELLIGENCE</div>
            <h1>Heart Disease Classification</h1>
            <p>
                Explore a machine-learning research project that
                classifies the presence or absence of heart disease
                from 13 clinical features using a trained Random
                Forest model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_heading(
        "Model at a glance",
        "Evaluation results from the selected model's test set."
    )

    row1 = st.columns(4)

    with row1[0]:
        metric_card("Accuracy", "85.00%", "Test set")

    with row1[1]:
        metric_card("Precision", "88.00%", "Test set")

    with row1[2]:
        metric_card("Recall", "78.57%", "Test set")

    with row1[3]:
        metric_card("F1-score", "83.02%", "Test set")

    st.write("")

    row2 = st.columns([1.1, 1])

    with row2[0]:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Selected model</div>
                <h2 style="margin-top:15px;">Random Forest</h2>
                <p class="muted">
                    An ensemble learning algorithm that combines
                    multiple decision trees to produce a classification.
                </p>
                <hr style="border-color:#E5ECF4;">
                <p><strong>Input features</strong> · 13</p>
                <p><strong>Task</strong> · Binary classification</p>
                <p><strong>Dataset</strong> · UCI Heart Disease</p>
                <p><strong>Framework</strong> · Scikit-learn / Streamlit</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with row2[1]:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Research objectives</div>
                <p class="muted">
                    This project demonstrates the application of
                    supervised machine learning to a clinical dataset.
                </p>
                <br>
                <p>✓ Prepare clinical features for model input</p>
                <p>✓ Compare classification algorithms</p>
                <p>✓ Evaluate predictions using classification metrics</p>
                <p>✓ Demonstrate a simple interactive application</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Ready to explore the model? Select Classifier in the "
        "navigation bar to enter sample feature values."
    )

# =========================================================
# 8. CLASSIFIER PAGE
# =========================================================

elif page == "Classifier":

    section_heading(
        "Heart Disease Classifier",
        "Enter feature values using the coding conventions "
        "of the original UCI dataset."
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Before you begin</div>
            <div class="muted">
                Use this form for educational testing and experimentation.
                Values should follow the original dataset's definitions.
                They should not be treated as medical measurements or advice.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("heart_disease_form", clear_on_submit=False):

        st.markdown("#### 01 · Demographic information")
        demographic1, demographic2, demographic3 = st.columns(3)

        with demographic1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=50
            )

        with demographic2:
            sex = st.selectbox(
                "Sex (dataset code)",
                [0, 1],
                format_func=lambda x: (
                    "0 — Female" if x == 0 else "1 — Male"
                )
            )

        with demographic3:
            cp = st.selectbox(
                "Chest pain type (cp)",
                [1, 2, 3, 4],
                format_func=lambda x: f"Category {x}"
            )

        st.divider()

        st.markdown("#### 02 · Resting measurements")
        rest1, rest2, rest3 = st.columns(3)

        with rest1:
            trestbps = st.number_input(
                "Resting blood pressure (trestbps)",
                min_value=50,
                max_value=300,
                value=120
            )

        with rest2:
            chol = st.number_input(
                "Serum cholesterol (chol)",
                min_value=50,
                max_value=700,
                value=200
            )

        with rest3:
            fbs = st.selectbox(
                "Fasting blood sugar above 120 mg/dl (fbs)",
                [0, 1],
                format_func=lambda x: (
                    "0 — No" if x == 0 else "1 — Yes"
                )
            )

        st.divider()

        st.markdown("#### 03 · Cardiac test features")
        test1, test2, test3 = st.columns(3)

        with test1:
            restecg = st.selectbox(
                "Resting ECG result (restecg)",
                [0, 1, 2],
                format_func=lambda x: f"Category {x}"
            )

        with test2:
            thalach = st.number_input(
                "Maximum heart rate (thalach)",
                min_value=50,
                max_value=250,
                value=150
            )

        with test3:
            exang = st.selectbox(
                "Exercise-induced angina (exang)",
                [0, 1],
                format_func=lambda x: (
                    "0 — No" if x == 0 else "1 — Yes"
                )
            )

        st.divider()

        st.markdown("#### 04 · Additional clinical features")
        extra1, extra2, extra3, extra4 = st.columns(4)

        with extra1:
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        with extra2:
            slope = st.selectbox(
                "ST segment slope (slope)",
                [1, 2, 3],
                format_func=lambda x: f"Category {x}"
            )

        with extra3:
            ca = st.selectbox(
                "Major vessels (ca)",
                [0, 1, 2, 3],
                format_func=lambda x: f"{x} vessel(s)"
            )

        with extra4:
            thal = st.selectbox(
                "Thal category (thal)",
                [3, 6, 7],
                format_func=lambda x: f"Category {x}"
            )

        st.write("")

        submitted = st.form_submit_button(
            "Run Classification →",
            use_container_width=True
        )

    # Process submitted input
    if submitted:

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

            st.write("")

            section_heading(
                "Classification output",
                "The following is the trained model's output "
                "for the values entered above."
            )

            if prediction == 1:

                st.markdown(
                    """
                    <div class="result-positive">
                        <div class="result-title" style="color:#A52A2A;">
                            Model output: Class 1
                        </div>
                        <p style="color:#7F3434;">
                            The model classified the submitted features
                            into the dataset's positive class for
                            heart disease.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif prediction == 0:

                st.markdown(
                    """
                    <div class="result-negative">
                        <div class="result-title" style="color:#087F70;">
                            Model output: Class 0
                        </div>
                        <p style="color:#245E54;">
                            The model classified the submitted features
                            into the dataset's negative class for
                            heart disease.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.warning(
                    f"The model returned an unexpected class: {prediction}"
                )

            st.caption(
                "This is a machine-learning classification, not a "
                "medical diagnosis. A result in either class does not "
                "establish whether a person has or does not have "
                "heart disease. Seek qualified medical advice for "
                "health concerns."
            )

        except Exception as error:
            st.error(
                "Prediction failed. Verify that the input feature "
                "names, order, and data types match the model used "
                "during training."
            )
            st.exception(error)

# =========================================================
# 9. MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    section_heading(
        "Model Performance",
        "Reported evaluation results for the selected Random Forest model."
    )

    performance1 = st.columns(3)

    with performance1[0]:
        metric_card("Accuracy", "85.00%", "Correct classifications")

    with performance1[1]:
        metric_card("Precision", "88.00%", "Positive predictions")

    with performance1[2]:
        metric_card("Recall", "78.57%", "Actual positive cases detected")

    performance2 = st.columns(3)

    with performance2[0]:
        metric_card("F1-score", "83.02%", "Precision-recall balance")

    with performance2[1]:
        metric_card("F2-score", "80.29%", "Greater weight on recall")

    with performance2[2]:
        metric_card("ROC-AUC", "94.08%", "Discrimination metric")

    st.write("")

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Understanding the metrics</div>
            <p class="muted">
                Each metric describes a different aspect of model
                performance. No single metric establishes clinical safety.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Accuracy"):
        st.write(
            "The proportion of all test examples that the model "
            "classified correctly."
        )

    with st.expander("Precision"):
        st.write(
            "Among examples predicted as positive, the proportion "
            "that were actually positive."
        )

    with st.expander("Recall"):
        st.write(
            "Among actual positive examples, the proportion that "
            "the model correctly identified."
        )

    with st.expander("F1-score and F2-score"):
        st.write(
            "F1 balances precision and recall. F2 gives more weight "
            "to recall than precision. Both summarize performance "
            "using the selected positive class."
        )

    with st.expander("ROC-AUC"):
        st.write(
            "ROC-AUC measures how well the model separates the two "
            "classes across classification thresholds. It is not "
            "the same as accuracy or an individual prediction's "
            "probability of disease."
        )

    st.warning(
        "These values are the reported evaluation metrics. Confirm "
        "they were calculated from the same test split and model "
        "version used for this deployment. They do not establish "
        "performance on new populations or in clinical practice."
    )

# =========================================================
# 10. ABOUT PAGE
# =========================================================

elif page == "About":

    section_heading(
        "About the Project",
        "A student machine-learning project exploring clinical "
        "data classification."
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Project overview</div>
            <p class="muted">
                This application demonstrates how a trained supervised
                learning model can classify records from the UCI Heart
                Disease dataset using 13 input features.
            </p>
            <br>
            <strong>Selected deployment model</strong>
            <p class="muted">Random Forest Classifier</p>
            <strong>Dataset source</strong>
            <p class="muted">UCI Machine Learning Repository</p>
            <strong>Application framework</strong>
            <p class="muted">Python, Streamlit, Pandas and Joblib</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Models explored")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Logistic Regression</div>
                <p class="muted">
                    A linear classification algorithm used as a
                    baseline model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with model_col2:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Random Forest</div>
                <p class="muted">
                    An ensemble of decision trees selected for
                    this demonstration.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with model_col3:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Neural Network (MLP)</div>
                <p class="muted">
                    A feed-forward neural network explored during
                    model comparison.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Important limitations")

    st.write(
        "- The application is for educational and research demonstration."
    )
    st.write(
        "- The dataset may not represent every population or clinical setting."
    )
    st.write(
        "- Model performance can change on unseen data."
    )
    st.write(
        "- The model output must not replace professional assessment, "
        "medical testing, or clinical judgment."
    )

    st.markdown("### Data and model integrity")

    st.write(
        "The model's feature order, preprocessing steps, and category "
        "encodings must match the original training pipeline. This "
        "application passes the 13 named features directly to the "
        "saved model; if your training pipeline requires scaling, "
        "imputation, or other transformations, save and load the "
        "complete fitted pipeline instead."
    )

# =========================================================
# 11. FOOTER
# =========================================================

render_footer()

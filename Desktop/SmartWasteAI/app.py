
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartWaste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
   MAIN BACKGROUND
========================================================= */

.stApp {

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(16,185,129,0.18),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 20%,
            rgba(34,197,94,0.14),
            transparent 28%
        ),

        radial-gradient(
            circle at 80% 90%,
            rgba(5,150,105,0.12),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #f0fdf4 0%,
            #f8fafc 45%,
            #ecfdf5 100%
        );

    min-height: 100vh;
}


/* =========================================================
   FLOATING ECO DECORATIONS
========================================================= */

.stApp::before {

    content: "♻️    🌿    🍃    ♻️    🌱    🍃";

    position: fixed;

    top: 8%;
    left: 3%;

    width: 100%;

    font-size: 55px;

    letter-spacing: 55px;

    opacity: 0.045;

    pointer-events: none;

    z-index: 0;

    transform: rotate(-8deg);
}


.stApp::after {

    content: "🌱     ♻️     🍃     🌿";

    position: fixed;

    bottom: 5%;
    right: 2%;

    font-size: 65px;

    letter-spacing: 45px;

    opacity: 0.04;

    pointer-events: none;

    z-index: 0;

    transform: rotate(8deg);
}


/* =========================================================
   MAIN CONTENT
========================================================= */

.block-container {

    position: relative;

    z-index: 1;

    padding-top: 2rem;

    padding-bottom: 3rem;

    max-width: 1400px;
}


/* =========================================================
   HERO SECTION
========================================================= */

.hero {

    padding: 40px;

    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(6,78,59,0.98),
            rgba(4,120,87,0.95),
            rgba(16,185,129,0.90)
        );

    color: white;

    margin-bottom: 28px;

    box-shadow:
        0 20px 50px rgba(4,120,87,0.22);

    border:
        1px solid rgba(255,255,255,0.18);

    position: relative;

    overflow: hidden;
}


.hero::after {

    content: "♻️";

    position: absolute;

    right: 45px;

    top: 5px;

    font-size: 130px;

    opacity: 0.10;

    transform: rotate(15deg);
}


.hero h1 {

    font-size: 46px;

    font-weight: 800;

    margin: 0;

    color: white;
}


.hero p {

    font-size: 18px;

    opacity: 0.92;

    margin-bottom: 5px;
}


/* =========================================================
   GLASS CARDS
========================================================= */

.card {

    background:
        rgba(255,255,255,0.84);

    backdrop-filter:
        blur(14px);

    -webkit-backdrop-filter:
        blur(14px);

    padding: 26px;

    border-radius: 22px;

    border:
        1px solid rgba(255,255,255,0.75);

    box-shadow:
        0 10px 35px rgba(15,23,42,0.07);

    margin-bottom: 20px;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}


.card:hover {

    transform:
        translateY(-4px);

    box-shadow:
        0 18px 45px rgba(15,23,42,0.12);
}


/* =========================================================
   KPI CARDS
========================================================= */

.kpi {

    background:
        rgba(255,255,255,0.90);

    backdrop-filter:
        blur(12px);

    padding: 25px;

    border-radius: 20px;

    border:
        1px solid rgba(255,255,255,0.8);

    box-shadow:
        0 8px 30px rgba(15,23,42,0.07);

    text-align: center;

    transition:
        transform 0.25s ease;
}


.kpi:hover {

    transform:
        translateY(-5px);
}


.kpi-value {

    font-size: 34px;

    font-weight: 800;

    color: #047857;
}


.kpi-label {

    color: #667085;

    font-size: 14px;

    margin-top: 6px;
}


/* =========================================================
   PREDICTION CARD
========================================================= */

.prediction {

    background:
        linear-gradient(
            135deg,
            rgba(236,253,245,0.96),
            rgba(209,250,229,0.92)
        );

    padding: 32px;

    border-radius: 24px;

    border:
        1px solid #a7f3d0;

    text-align: center;

    box-shadow:
        0 12px 35px rgba(5,150,105,0.12);
}


.prediction-label {

    font-size: 15px;

    color: #475467;
}


.prediction-class {

    font-size: 40px;

    font-weight: 800;

    color: #047857;

    margin: 8px 0;
}


.prediction-confidence {

    font-size: 21px;

    font-weight: 700;

    color: #065f46;
}


/* =========================================================
   CATEGORY BADGES
========================================================= */

.category {

    display: inline-block;

    padding: 9px 16px;

    margin: 5px;

    border-radius: 25px;

    background:
        rgba(236,253,245,0.90);

    color: #047857;

    border:
        1px solid #a7f3d0;

    font-weight: 600;

    transition:
        transform 0.2s ease,
        background 0.2s ease;
}


.category:hover {

    transform:
        translateY(-3px);

    background:
        #d1fae5;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #022c22,
            #064e3b,
            #065f46
        );
}


section[data-testid="stSidebar"] * {

    color: white;
}


/* =========================================================
   FILE UPLOADER
========================================================= */

[data-testid="stFileUploader"] {

    background:
        rgba(255,255,255,0.88);

    backdrop-filter:
        blur(10px);

    border-radius: 18px;

    padding: 12px;

    border:
        1px dashed #10b981;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.06);
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {

    border-radius: 12px;

    border: none;

    background:
        linear-gradient(
            135deg,
            #047857,
            #10b981
        );

    color: white;

    font-weight: 700;

    padding: 10px 22px;

    box-shadow:
        0 5px 15px rgba(4,120,87,0.20);

    transition:
        transform 0.2s ease;
}


.stButton > button:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 8px 20px rgba(4,120,87,0.30);
}


/* =========================================================
   HEADINGS
========================================================= */

h1,
h2,
h3 {

    color: #172033;

    font-weight: 750;
}


/* =========================================================
   HIDE STREAMLIT DEFAULT ELEMENTS
========================================================= */

#MainMenu {

    visibility: hidden;
}


footer {

    visibility: hidden;
}



/* =========================================================
   RESPONSIVE LAYOUT + READABILITY POLISH
   ========================================================= */
[data-testid="stMainBlockContainer"],
.block-container {
    width: 100% !important;
    max-width: 1500px !important;
    padding-left: clamp(1rem, 3vw, 3rem) !important;
    padding-right: clamp(1rem, 3vw, 3rem) !important;
    padding-top: 1.6rem !important;
    padding-bottom: 3rem !important;
}

[data-testid="stAppViewContainer"] .main {
    min-width: 0;
}

/* Keep body copy readable at normal browser zoom. */
[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stCaptionContainer"] {
    font-size: 1rem;
    line-height: 1.6;
    color: #263548;
}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #172033 !important;
    line-height: 1.25;
    letter-spacing: -0.02em;
}

.hero h1,
.hero p {
    color: #ffffff !important;
}
.hero {
    padding: clamp(1.5rem, 3vw, 2.8rem) !important;
    margin-bottom: 1.5rem !important;
}
.hero h1 {
    font-size: clamp(2rem, 3.2vw, 2.9rem) !important;
}
.hero p {
    font-size: clamp(1rem, 1.2vw, 1.15rem) !important;
}

.card, .kpi {
    color: #1f2937 !important;
}
.card p, .card li, .card span, .card strong,
.card h1, .card h2, .card h3, .kpi-label {
    color: #263548 !important;
}
.kpi-value {
    color: #047857 !important;
    font-size: clamp(1.7rem, 2.4vw, 2.2rem) !important;
}
.kpi-label {
    font-size: 0.95rem !important;
}
.prediction-label { color: #475467 !important; }
.prediction-class { color: #047857 !important; }
.prediction-confidence { color: #065f46 !important; }

/* Make widgets and charts use the available content width. */
[data-testid="stFileUploader"],
[data-testid="stCameraInput"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    width: 100% !important;
}
[data-testid="stMetricValue"] {
    font-size: clamp(1.4rem, 2vw, 2rem) !important;
}

/* Sidebar intentionally retains its dark-green / white treatment. */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #ffffff !important;
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"], .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .hero::after { font-size: 4rem; right: 1rem; }
    .card { padding: 1.2rem !important; }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD CNN MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        str(BASE_DIR / "cnn_model.keras")
    )

    return model


model = load_model()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    str(BASE_DIR / "class_names.json"),
    "r"
) as file:

    loaded_classes = json.load(file)


# Handle either JSON list or dictionary

if isinstance(loaded_classes, dict):

    # Example:
    # {"battery": 0, "plastic": 8}

    class_names = [
        name
        for name, index
        in sorted(
            loaded_classes.items(),
            key=lambda x: x[1]
        )
    ]

else:

    class_names = loaded_classes


# ============================================================
# MODEL PERFORMANCE
# ============================================================

TEST_ACCURACY = 71.60

TEST_LOSS = 0.8915

PRECISION = 0.74

RECALL = 0.72

F1_SCORE = 0.72

TEST_IMAGES = 4698


# ============================================================
# CLASS-WISE F1 SCORE
# ============================================================

class_f1 = {

    "battery": 0.72,

    "biological": 0.79,

    "brown-glass": 0.76,

    "cardboard": 0.67,

    "clothes": 0.88,

    "green-glass": 0.86,

    "metal": 0.61,

    "paper": 0.62,

    "plastic": 0.66,

    "shoes": 0.73,

    "trash": 0.53,

    "white-glass": 0.57
}


# ============================================================
# DISPOSAL RECOMMENDATIONS
# ============================================================

recommendations = {

    "battery":
        "🔋 Dispose of batteries at an authorized e-waste collection point.",

    "biological":
        "🌱 Place biological waste in the organic or composting waste bin.",

    "brown-glass":
        "🍾 Place brown glass in the glass recycling bin.",

    "cardboard":
        "📦 Flatten cardboard and place it in the paper/cardboard recycling bin.",

    "clothes":
        "👕 Donate reusable clothes or send them for textile recycling.",

    "green-glass":
        "🍾 Place green glass in the glass recycling bin.",

    "metal":
        "🔩 Place metal in the metal recycling collection.",

    "paper":
        "📄 Place clean paper in the paper recycling bin.",

    "plastic":
        "♻️ Place recyclable plastic in the plastic recycling bin.",

    "shoes":
        "👟 Donate reusable shoes or send them to textile/shoe recycling.",

    "trash":
        "🗑️ Place this item in the general waste bin.",

    "white-glass":
        "🥛 Place white glass in the glass recycling bin."
}


# ============================================================
# SESSION HISTORY
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ♻️ SmartWaste AI"
)

st.sidebar.caption(
    "Smart Campus Waste Management"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [
        "🏠 Dashboard",
        "🔍 Classify Waste",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 🧠 Model Information"
)

st.sidebar.write(
    "Model: CNN"
)

st.sidebar.write(
    "Input: 224 × 224"
)

st.sidebar.write(
    "Categories: 12"
)

st.sidebar.write(
    "Test Accuracy: 71.60%"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-powered waste classification"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">

        <h1>♻️ SmartWaste AI</h1>

        <p>
        AI-Powered Smart Campus Waste Management
        </p>

        <p>
        Automatically classify waste using
        Convolutional Neural Networks and promote
        smarter waste segregation.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            """
            <div class="kpi">

            <div class="kpi-value">
            71.60%
            </div>

            <div class="kpi-label">
            Test Accuracy
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="kpi">

            <div class="kpi-value">
            0.72
            </div>

            <div class="kpi-label">
            Weighted F1 Score
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="kpi">

            <div class="kpi-value">
            4,698
            </div>

            <div class="kpi-label">
            Test Images
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            """
            <div class="kpi">

            <div class="kpi-value">
            12
            </div>

            <div class="kpi-label">
            Waste Categories
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    col1, col2 = st.columns([1.5, 1])


    with col1:

        st.markdown(
            """
            <div class="card">

            <h2>🌍 Smart Waste Management</h2>

            <p>
            Improper waste segregation is a major challenge
            in educational campuses. SmartWaste AI uses
            computer vision and deep learning to identify
            waste categories from images.
            </p>

            <p>
            The system helps users understand how waste
            should be categorized and provides a recommended
            disposal method.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">

            <h3>🚀 Project Pipeline</h3>

            <p>📷 Image Input</p>

            <p>⬇️</p>

            <p>🧠 CNN Processing</p>

            <p>⬇️</p>

            <p>🎯 Classification</p>

            <p>⬇️</p>

            <p>♻️ Disposal Recommendation</p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # CATEGORIES
    # --------------------------------------------------------

    st.markdown(
        "### 🗂️ Supported Waste Categories"
    )

    category_html = ""

    for category in class_names:

        category_html += (
            f'<span class="category">'
            f'{category}'
            f'</span>'
        )

    st.markdown(
        category_html,
        unsafe_allow_html=True
    )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

        <h3>🧠 CNN Model Summary</h3>

        <p>
        The trained CNN receives an image of size
        <strong>224 × 224 pixels</strong> and classifies
        it into one of <strong>12 waste categories</strong>.
        </p>

        <p>
        The current model achieved a test accuracy of
        <strong>71.60%</strong> and weighted F1-score of
        <strong>0.72</strong>.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CLASSIFY WASTE
# ============================================================

elif page == "🔍 Classify Waste":

    st.markdown(
        """
        <div class="hero">

        <h1>🔍 Waste Classification</h1>

        <p>
        Upload a waste image or capture one using your camera.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    upload_col, camera_col = st.columns(2)


    with upload_col:

        uploaded_file = st.file_uploader(

            "📁 Upload Waste Image",

            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )


    with camera_col:

        camera_file = st.camera_input(
            "📷 Capture Waste Image"
        )


    # Select input

    if uploaded_file is not None:

        image_file = uploaded_file

    elif camera_file is not None:

        image_file = camera_file

    else:

        image_file = None


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if image_file is not None:

        image = Image.open(
            image_file
        ).convert("RGB")


        st.divider()


        image_col, result_col = st.columns(
            [1, 1]
        )


        # ----------------------------------------------------
        # INPUT IMAGE
        # ----------------------------------------------------

        with image_col:

            st.markdown(
                "### 📷 Input Image"
            )

            st.image(
                image,
                use_container_width=True
            )


        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        resized_image = image.resize(
            (224, 224)
        )


        image_array = np.array(
            resized_image,
            dtype=np.float32
        )


        # IMPORTANT:
        # Same normalization used by the application

        image_array = image_array / 255.0


        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # CNN PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(prediction)
        )


        predicted_class = class_names[
            predicted_index
        ]


        confidence = float(
            prediction[predicted_index] * 100
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with result_col:

            st.markdown(
                "### 🤖 AI Prediction"
            )


            st.markdown(
                f"""
                <div class="prediction">

                <div class="prediction-label">
                Predicted Waste Category
                </div>

                <div class="prediction-class">
                ♻️ {predicted_class}
                </div>

                <div class="prediction-confidence">
                {confidence:.2f}% Confidence
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.progress(
                float(
                    prediction[predicted_index]
                )
            )


        st.markdown("<br>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # CONFIDENCE MESSAGE
        # ----------------------------------------------------

        if confidence >= 80:

            st.success(
                "🟢 High-confidence prediction"
            )

        elif confidence >= 60:

            st.info(
                "🟡 Moderate-confidence prediction"
            )

        else:

            st.warning(
                "🔴 Low-confidence prediction. "
                "Try a clearer image with the waste "
                "item centered and well illuminated."
            )


        # ----------------------------------------------------
        # DISPOSAL RECOMMENDATION
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="card">

            <h3>♻️ Recommended Disposal</h3>

            """,
            unsafe_allow_html=True
        )


        st.write(
            recommendations.get(
                predicted_class,
                "Dispose of the item appropriately."
            )
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # TOP 3 PREDICTIONS
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Top 3 Predictions"
        )


        top_indices = np.argsort(
            prediction
        )[-3:][::-1]


        top_data = []


        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            top_data.append({

                "Rank":
                    rank,

                "Category":
                    class_names[index],

                "Confidence":
                    round(
                        float(
                            prediction[index] * 100
                        ),
                        2
                    )
            })


        top_df = pd.DataFrame(
            top_data
        )


        # Bar chart

        chart_df = top_df[
            [
                "Category",
                "Confidence"
            ]
        ].set_index(
            "Category"
        )


        st.bar_chart(
            chart_df
        )


        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # SAVE SESSION HISTORY
        # ----------------------------------------------------

        st.session_state.history.append({

            "Prediction":
                predicted_class,

            "Confidence":
                round(
                    confidence,
                    2
                )
        })


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        """
        <div class="hero">

        <h1>📊 Model Analytics</h1>

        <p>
        Performance analysis of the trained CNN model
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Test Accuracy",
            "71.60%"
        )


    with c2:

        st.metric(
            "Test Loss",
            "0.8915"
        )


    with c3:

        st.metric(
            "Precision",
            "0.74"
        )


    with c4:

        st.metric(
            "Recall",
            "0.72"
        )


    st.divider()


    # --------------------------------------------------------
    # F1 SCORE CHART
    # --------------------------------------------------------

    st.subheader(
        "📈 F1 Score by Waste Category"
    )


    f1_df = pd.DataFrame({

        "Category":
            list(class_f1.keys()),

        "F1 Score":
            list(class_f1.values())
    })


    st.bar_chart(
        f1_df.set_index(
            "Category"
        )
    )


    # --------------------------------------------------------
    # MODEL METRICS TABLE
    # --------------------------------------------------------

    st.subheader(
        "📋 Overall Model Evaluation"
    )


    metrics_df = pd.DataFrame({

        "Metric": [

            "Test Accuracy",

            "Test Loss",

            "Precision",

            "Recall",

            "Weighted F1 Score"

        ],

        "Value": [

            "71.60%",

            "0.8915",

            "0.74",

            "0.72",

            "0.72"

        ]
    })


    st.dataframe(

        metrics_df,

        use_container_width=True,

        hide_index=True
    )


    # --------------------------------------------------------
    # CLASS PERFORMANCE TABLE
    # --------------------------------------------------------

    st.subheader(
        "🗂️ Class-wise F1 Performance"
    )


    class_df = pd.DataFrame({

        "Waste Category":
            list(class_f1.keys()),

        "F1 Score":
            list(class_f1.values())
    })


    st.dataframe(

        class_df,

        use_container_width=True,

        hide_index=True
    )


    # --------------------------------------------------------
    # SESSION PREDICTION HISTORY
    # --------------------------------------------------------

    st.subheader(
        "🕒 Current Session Predictions"
    )


    if len(
        st.session_state.history
    ) > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )


        st.dataframe(

            history_df,

            use_container_width=True,

            hide_index=True
        )

    else:

        st.info(
            "No predictions have been made "
            "during this session."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        """
        <div class="hero">

        <h1>ℹ️ About SmartWaste AI</h1>

        <p>
        Waste Classification for Smart Campus Waste Management
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PROBLEM
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

        <h2>🌍 Real-World Problem</h2>

        <p>
        Improper waste segregation is a major environmental
        and operational challenge in educational campuses.
        Different waste materials require different disposal
        and recycling methods.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SOLUTION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

        <h2>💡 Proposed Solution</h2>

        <p>
        SmartWaste AI uses a Convolutional Neural Network
        to automatically classify waste from an image.
        The application then provides a disposal recommendation
        to the user.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

        <h2>🛠️ Technologies Used</h2>

        <p>🐍 Python</p>

        <p>🧠 TensorFlow / Keras</p>

        <p>👁️ Convolutional Neural Network</p>

        <p>📊 NumPy / Pandas / Matplotlib</p>

        <p>🌐 Streamlit</p>

        <p>🖼️ PIL</p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

        <h2>🔄 Project Workflow</h2>

        <p>
        Data Collection
        →
        Data Cleaning
        →
        EDA
        →
        Train/Validation/Test Split
        →
        CNN Training
        →
        Evaluation
        →
        Best Model
        →
        Streamlit Application
        →
        New Image
        →
        Prediction
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

        <h2>🏆 Current Model Performance</h2>

        <p>
        Test Accuracy:
        <strong>71.60%</strong>
        </p>

        <p>
        Weighted F1 Score:
        <strong>0.72</strong>
        </p>

        <p>
        Test Images:
        <strong>4,698</strong>
        </p>

        <p>
        Waste Categories:
        <strong>12</strong>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CATEGORIES
    # --------------------------------------------------------

    st.subheader(
        "🗂️ Supported Categories"
    )


    category_html = ""


    for category in class_names:

        category_html += (

            f'<span class="category">'
            f'{category}'
            f'</span>'

        )


    st.markdown(

        category_html,

        unsafe_allow_html=True
    )


    st.markdown("---")


    st.caption(
        "SmartWaste AI | Deep Learning Project | CNN"
    )
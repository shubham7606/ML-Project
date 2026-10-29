import streamlit as st
import pandas as pd
import joblib
from scipy.special import boxcox as boxcox_transform

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="MobileIQ | Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LOAD TRAINED OBJECTS
# --------------------------------------------------
@st.cache_resource
def load_objects():
    boxcox_lambdas = joblib.load("boxcox_lambdas.pkl")
    model = joblib.load("gradient_boosting_model.pkl")
    scaler = joblib.load("standard_scalar.pkl")
    return boxcox_lambdas, model, scaler

boxcox_lambdas, model, scaler = load_objects()

# Preserve exact training feature names and order
feature_columns = list(scaler.feature_names_in_)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(180deg, #f2f5ff 0%, #f8f9fd 50%, #ffffff 100%);
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #202b4a;
    }

    .hero {
        background: linear-gradient(120deg, #172554 0%, #3730a3 55%, #6366f1 100%);
        padding: 38px;
        border-radius: 24px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 35px rgba(55, 48, 163, 0.18);
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: clamp(29px, 4vw, 43px);
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }

    .hero p {
        color: #e0e7ff;
        font-size: 15px;
        line-height: 1.8;
        margin: 12px 0 0;
        max-width: 700px;
    }

    .section-heading {
        color: #202b4a;
        font-size: 21px;
        font-weight: 800;
        margin: 12px 0 5px;
    }

    .section-description {
        color: #7a839b;
        font-size: 14px;
        margin-bottom: 18px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid #e3e7f3;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(35, 46, 85, 0.045);
    }

    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #34405e !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        background: linear-gradient(100deg, #3730a3, #6366f1);
        color: white;
        font-size: 16px;
        font-weight: 700;
        transition: all 0.2s ease;
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.2);
    }

    .stButton > button:hover {
        background: linear-gradient(100deg, #312e81, #4f46e5);
        color: white;
        border: none;
        transform: translateY(-2px);
    }

    .metric-card {
        background: white;
        border: 1px solid #e3e7f3;
        border-radius: 15px;
        padding: 16px 10px;
        text-align: center;
        min-height: 91px;
    }

    .metric-label {
        font-size: 12px;
        color: #7a839b;
        font-weight: 600;
    }

    .metric-value {
        font-size: 21px;
        color: #3730a3;
        font-weight: 800;
        margin-top: 7px;
    }

    .result-card {
        background: linear-gradient(135deg, #eef2ff, #f5f3ff);
        border: 1px solid #c7d2fe;
        border-radius: 22px;
        padding: 30px 20px;
        text-align: center;
        margin-top: 24px;
    }

    .result-label {
        color: #6366a6;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .result-price {
        color: #312e81;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(32px, 5vw, 46px);
        font-weight: 800;
        margin: 12px 0;
    }

    .result-note {
        color: #656b89;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #8991a8;
        font-size: 12px;
        margin-top: 38px;
        padding-top: 18px;
        border-top: 1px solid #e3e7f3;
        line-height: 1.9;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 640px) {
        .hero {
            padding: 25px 21px;
        }

        .block-container {
            padding: 1rem 1rem 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-tag">📱 SMART DEVICE ANALYTICS</div>
    <h1>Discover Your Mobile's<br>Estimated Price.</h1>
    <p>
        Configure your smartphone specifications and let machine learning
        estimate its price based on the patterns learned from your training data.
        Adjust the hardware details below to get started.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROCESSOR & DISPLAY
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">⚙️ Processor & Display</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">Configure the performance and display specifications.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### ⚖️ Device Weight")
        weight = st.slider(
            "Weight (grams)", 100, 900, 150,
            help="Total weight of the mobile device in grams."
        )

with col2:
    with st.container(border=True):
        st.markdown("### 🖥️ Screen Size")
        resolution = st.slider(
            "Resolution / screen size (inches)",
            1.0, 13.0, 1.0, 0.1,
            help="Use the same screen-size definition as your training dataset."
        )

with col3:
    with st.container(border=True):
        st.markdown("### 🔎 Pixel Density")
        ppi = st.number_input(
            "Pixels per inch (PPI)",
            100, 900, 200, step=1
        )

col4, col5 = st.columns(2)

with col4:
    with st.container(border=True):
        st.markdown("### 🧠 CPU Cores")
        cpu_cores = st.selectbox(
            "Number of CPU cores",
            [1, 2, 3, 4, 5, 6, 7, 8]
        )

with col5:
    with st.container(border=True):
        st.markdown("### 🚀 CPU Frequency")
        cpu_freq = st.slider(
            "CPU frequency (GHz)",
            0.5, 3.0, 1.0, 0.1
        )

# --------------------------------------------------
# MEMORY & STORAGE
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">💾 Memory & Storage</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">Choose the RAM and internal storage capacity.</div>',
    unsafe_allow_html=True
)

col6, col7 = st.columns(2)

with col6:
    with st.container(border=True):
        st.markdown("### 💿 Internal Storage")
        internal_storage = st.selectbox(
            "Storage capacity (GB)",
            [4, 8, 16, 32, 64, 128]
        )

with col7:
    with st.container(border=True):
        st.markdown("### 🧩 RAM")
        ram = st.selectbox(
            "RAM capacity (GB)",
            [1, 1.5, 2, 3, 4, 6]
        )

# --------------------------------------------------
# CAMERA & BATTERY
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">📸 Camera & Battery</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">Complete your device profile with camera and battery details.</div>',
    unsafe_allow_html=True
)

col8, col9, col10 = st.columns(3)

with col8:
    with st.container(border=True):
        st.markdown("### 📷 Rear Camera")
        rear_camera = st.slider(
            "Rear camera (MP)", 1, 50, 15
        )

with col9:
    with st.container(border=True):
        st.markdown("### 🤳 Front Camera")
        front_camera = st.slider(
            "Front camera (MP)", 1, 50, 15
        )

with col10:
    with st.container(border=True):
        st.markdown("### 🔋 Battery")
        battery = st.slider(
            "Battery capacity (mAh)",
            800, 10000, 3000, step=100
        )

# --------------------------------------------------
# DEVICE SUMMARY
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">📋 Device Overview</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">Review your selected hardware specifications.</div>',
    unsafe_allow_html=True
)

summary = [
    ("Weight", f"{weight} g"),
    ("Screen", f"{resolution:.1f} in"),
    ("RAM", f"{ram} GB"),
    ("Storage", f"{internal_storage} GB"),
    ("Battery", f"{battery:,} mAh")
]

summary_cols = st.columns(5)

for col, (label, value) in zip(summary_cols, summary):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
predict_col, reset_col = st.columns([3, 1])

with predict_col:
    predict_clicked = st.button(
        "✨  Predict Mobile Price",
        use_container_width=True
    )

with reset_col:
    reset_clicked = st.button(
        "↻ Reset App",
        use_container_width=True
    )

if reset_clicked:
    st.session_state.clear()
    st.rerun()

if predict_clicked:
    try:
        raw_values = {
            "weight": weight,
            "resoloution": resolution,
            "ppi": ppi,
            "cpu core": cpu_cores,
            "cpu freq": cpu_freq,
            "internal mem": internal_storage,
            "ram": ram,
            "RearCam": rear_camera,
            "Front_Cam": front_camera,
            "battery": battery,
        }

        missing_columns = [
            col for col in feature_columns if col not in raw_values
        ]

        if missing_columns:
            raise ValueError(
                f"Missing model features: {missing_columns}"
            )

        input_data = pd.DataFrame(
            [[raw_values[col] for col in feature_columns]],
            columns=feature_columns
        )

        # Apply the same Box-Cox transformations as training
        for column, fitted_lambda in boxcox_lambdas.items():
            if column not in input_data.columns:
                raise ValueError(
                    f"Box-Cox feature '{column}' is missing."
                )

            input_data[column] = boxcox_transform(
                input_data[column].to_numpy() + 1,
                fitted_lambda
            )

        # Apply the fitted scaler
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = float(model.predict(input_scaled)[0])

        st.session_state["predicted_price"] = prediction

    except Exception as e:
        st.error(f"Prediction failed: {e}")

if "predicted_price" in st.session_state:
    prediction = st.session_state["predicted_price"]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated Mobile Price</div>
            <div class="result-price">{prediction:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )



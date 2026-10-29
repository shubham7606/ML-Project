import pandas as pd
import joblib as jd
import streamlit as st


st.set_page_config(
    page_title="CropSense | Smart Crop Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def load_models():
    model = jd.load("randomForest_crop_prediction.pkl")
    pt = jd.load("powerTransformer_crop_prediction.pkl")
    label_encoder = jd.load("label_encoder_crop_prediction.pkl")
    return model, pt, label_encoder

model, pt, label_encoder = load_models()

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(180deg, #f1f8f0 0%, #f8fbf6 45%, #ffffff 100%);
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #173d2b;
    }

    .hero {
        background: linear-gradient(120deg, #123d2a 0%, #206b43 65%, #41965c 100%);
        padding: 36px 38px;
        border-radius: 24px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 35px rgba(24, 83, 49, 0.15);
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 14px;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }

    .hero p {
        color: #e1f3e4;
        font-size: 16px;
        margin: 12px 0 0;
        line-height: 1.7;
    }

    .section-heading {
        color: #173d2b;
        font-size: 21px;
        font-weight: 800;
        margin: 12px 0 5px;
    }

    .section-description {
        color: #6b7c70;
        font-size: 14px;
        margin-bottom: 18px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid #e0eadf;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 4px 18px rgba(28, 66, 40, 0.045);
    }

    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label {
        color: #344d3c !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    div[data-testid="stSlider"] [data-testid="stTickBar"] {
        background: #dcebdd;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        color: white;
        background: linear-gradient(100deg, #1b7040, #2d9655);
        font-size: 16px;
        font-weight: 700;
        transition: all 0.2s ease;
        box-shadow: 0 5px 14px rgba(37, 126, 69, 0.18);
    }

    .stButton > button:hover {
        background: linear-gradient(100deg, #145b33, #247d45);
        transform: translateY(-2px);
        color: white;
        border: none;
    }

    .result-card {
        background: linear-gradient(135deg, #e2f5e6, #f5fbef);
        border: 1px solid #b9e2c1;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        margin-top: 24px;
    }

    .result-label {
        color: #42704e;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .result-crop {
        color: #155b31;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(28px, 4vw, 40px);
        font-weight: 800;
        margin: 10px 0;
        text-transform: capitalize;
    }

    .result-note {
        color: #4e6c55;
        font-size: 14px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e0eadf;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        min-height: 90px;
    }

    .metric-label {
        font-size: 12px;
        color: #718075;
        font-weight: 600;
    }

    .metric-value {
        font-size: 21px;
        color: #1c5835;
        font-weight: 800;
        margin-top: 6px;
    }

    .footer {
        text-align: center;
        color: #87968a;
        font-size: 12px;
        margin-top: 38px;
        padding-top: 18px;
        border-top: 1px solid #e5eee4;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 640px) {
        .hero {
            padding: 26px 22px;
        }
        .block-container {
            padding: 1rem 1rem 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="hero-tag">🌱 SMART AGRICULTURE</div>
    <h1>Find the Right Crop<br>for Your Land.</h1>
    <p>
        Enter your soil nutrients and local weather conditions.
        Our machine learning model will recommend a suitable crop
        based on the information you provide.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="section-heading">🌿 Soil Nutrients</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Provide the NPK values from your soil analysis.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### 🧪 Nitrogen")
        N = st.slider(
            "Nitrogen content (N)",
            min_value=0, max_value=140, value=40,
            help="Nitrogen concentration in soil."
        )

with col2:
    with st.container(border=True):
        st.markdown("### 🧪 Phosphorus")
        P = st.slider(
            "Phosphorus content (P)",
            min_value=0, max_value=145, value=40,
            help="Phosphorus concentration in soil."
        )

with col3:
    with st.container(border=True):
        st.markdown("### 🧪 Potassium")
        K = st.slider(
            "Potassium content (K)",
            min_value=0, max_value=205, value=40,
            help="Potassium concentration in soil."
        )

st.markdown('<div class="section-heading">🌦️ Weather & Soil Conditions</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Adjust the environmental values for your field.</div>',
    unsafe_allow_html=True
)

col4, col5 = st.columns(2)

with col4:
    with st.container(border=True):
        st.markdown("### 🌡️ Temperature")
        temp_typed = st.number_input(
            "Temperature (°C)",
            min_value=8.0, max_value=44.0,
            value=25.0, step=0.1
        )

        st.markdown("### 💧 Humidity")
        humidity = st.number_input(
            "Relative humidity (%)",
            min_value=14.0, max_value=100.0,
            value=25.0, step=0.1
        )

with col5:
    with st.container(border=True):
        st.markdown("### ⚗️ Soil pH")
        ph_value = st.slider(
            "pH level",
            min_value=3.0, max_value=10.0,
            value=4.0, step=0.1,
            help="Soil pH indicates acidity or alkalinity."
        )

        st.markdown("### 🌧️ Rainfall")
        rainfall = st.slider(
            "Rainfall (mm)",
            min_value=20.0, max_value=298.5,
            value=100.0, step=0.1
        )


st.markdown('<div class="section-heading">📋 Field Summary</div>',
            unsafe_allow_html=True)

summary_cols = st.columns(4)

summary = [
    ("Nitrogen", f"{N}"),
    ("Phosphorus", f"{P}"),
    ("Potassium", f"{K}"),
    ("Temperature", f"{temp_typed:.1f} °C")
]

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


predict_col, reset_col = st.columns([3, 1])

with predict_col:
    predict_clicked = st.button(
        "🌾  Predict Best Crop",
        use_container_width=True
    )

with reset_col:
    reset_clicked = st.button(
        "↻ Reset",
        use_container_width=True
    )

if reset_clicked:
    st.session_state.clear()
    st.rerun()

if predict_clicked:
    try:
        input_data = pd.DataFrame(
            [[N, P, K, temp_typed, humidity, ph_value, rainfall]],
            columns=FEATURES
        )

        transformed_data = pt.transform(input_data)
        prediction = model.predict(transformed_data)
        result = label_encoder.inverse_transform(prediction)[0]

        st.session_state["predicted_crop"] = str(result)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

if st.session_state.get("predicted_crop"):
    result = st.session_state["predicted_crop"]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Your Recommended Crop</div>
            <div class="result-crop">🌾 {result}</div>
            <div class="result-note">
                Recommendation generated from your entered soil
                and environmental values.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )





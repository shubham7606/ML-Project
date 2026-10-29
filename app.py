import streamlit as st
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="ML Model Hub",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# PATHS
# Folder structure:
#   app.py
#   images/mobile_results.png
#   images/crop_results.png
#   pages/1_mobile_price_prediction.py
#   pages/2_crop_prediction.py
# --------------------------------------------------
IMG_DIR = Path(__file__).parent / "images"


def show_screenshot(filename, caption, download_key):
    """Show a screenshot inside an expander, with a download button.
    Shows a friendly message if the image file is missing."""
    img_path = IMG_DIR / filename

    with st.expander("📊 View model comparison"):
        if img_path.exists():
            st.image(str(img_path), caption=caption, use_container_width=True)
            with open(img_path, "rb") as f:
                st.download_button(
                    "⬇ Download image",
                    data=f,
                    file_name=filename,
                    mime="image/png",
                    key=download_key,
                )
        else:
            st.info(f"Screenshot not found: images/{filename}")


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    /* ---------------------------------------------
       HIDE SIDEBAR
    --------------------------------------------- */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* ---------------------------------------------
       GLOBAL STYLES
    --------------------------------------------- */
    .stApp {
        background: linear-gradient(
            180deg,
            #f6f5ef 0%,
            #fbfaf6 55%,
            #ffffff 100%
        );
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #203a35;
    }

    /* ---------------------------------------------
       TOP NAVIGATION HEADER
    --------------------------------------------- */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 0 2px 24px;
        margin-bottom: 20px;
        border-bottom: 1px solid #e1e7dc;
    }

    .brand {
        color: #203a35;
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
    }

    .brand span {
        color: #3a7592;
    }

    .topbar-tag {
        color: #3a7592;
        background: #e5f0f6;
        padding: 8px 13px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
    }

    /* ---------------------------------------------
       HERO SECTION
    --------------------------------------------- */
    .hero {
        background: #3a7592;
        padding: 42px;
        border-radius: 24px;
        color: #ffffff;
        margin-bottom: 28px;
        box-shadow: 0 12px 35px rgba(58, 117, 146, 0.18);
        position: relative;
        overflow: hidden;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: clamp(32px, 4vw, 46px);
        font-weight: 800;
        line-height: 1.2;
        margin: 0;
    }

    .hero p {
        color: #eaf4f8;
        font-size: 15px;
        line-height: 1.8;
        margin: 15px 0 0;
        max-width: 720px;
    }

    .hero-decoration {
        float: right;
        font-size: 90px;
        opacity: 0.85;
        margin: 15px 25px 0 20px;
    }

    /* ---------------------------------------------
       STATISTICS
    --------------------------------------------- */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 18px;
        margin: 25px 0 36px;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e1e7dc;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 4px 18px rgba(32, 58, 53, 0.04);
    }

    .stat-icon {
        font-size: 25px;
        margin-bottom: 10px;
    }

    .stat-number {
        color: #3a7592;
        font-size: 30px;
        font-weight: 800;
        font-family: 'Manrope', sans-serif;
    }

    .stat-label {
        color: #78867c;
        font-size: 13px;
        margin-top: 4px;
    }

    /* ---------------------------------------------
       PROJECT SECTION HEADINGS
    --------------------------------------------- */
    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: #203a35;
        font-size: 25px;
        font-weight: 800;
        margin: 8px 0 6px;
    }

    .section-description {
        color: #78867c;
        font-size: 14px;
        margin-bottom: 24px;
    }

    /* ---------------------------------------------
       PROJECT CARDS
    --------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #e1e7dc;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 5px 22px rgba(32, 58, 53, 0.05);
        transition: none !important;
    }

    .project-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 64px;
        height: 64px;
        background: #e5f0f6;
        border-radius: 18px;
        font-size: 34px;
        margin-bottom: 20px;
    }

    .project-title {
        color: #203a35;
        font-family: 'Manrope', sans-serif;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .project-description {
        color: #78867c;
        font-size: 14px;
        line-height: 1.8;
        min-height: 75px;
    }

    .project-type {
        display: inline-block;
        margin-top: 14px;
        background: #e5f0f6;
        color: #315f78;
        padding: 7px 11px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 700;
    }

    /* ---------------------------------------------
       OPEN / PAGE LINK BUTTONS
       Keep the same color on hover, focus and click
    --------------------------------------------- */
    .stPageLink a,
    div[data-testid="stPageLink"] a {
        background-color: #3a7592 !important;
        border: 1px solid #3a7592 !important;
        color: #ffffff !important;
        border-radius: 11px !important;
        padding: 11px 16px;
        font-weight: 700 !important;
        text-decoration: none !important;
        box-shadow: none !important;
        outline: none !important;
        transition: none !important;
    }

    .stPageLink a:hover,
    .stPageLink a:focus,
    .stPageLink a:active,
    div[data-testid="stPageLink"] a:hover,
    div[data-testid="stPageLink"] a:focus,
    div[data-testid="stPageLink"] a:active {
        background-color: #3a7592 !important;
        border-color: #3a7592 !important;
        color: #ffffff !important;
        box-shadow: none !important;
        outline: none !important;
        filter: none !important;
    }

    /* Keep button text and icons white */
    div[data-testid="stPageLink"] a span,
    div[data-testid="stPageLink"] a p,
    div[data-testid="stPageLink"] a svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* ---------------------------------------------
       BOTTOM INFORMATION BANNER
    --------------------------------------------- */
    .bottom-banner {
        background: linear-gradient(
            120deg,
            #e5f0f6,
            #f7f3e5
        );
        border: 1px solid #d8e5ec;
        border-radius: 20px;
        padding: 25px 28px;
        margin-top: 32px;
    }

    .bottom-title {
        color: #315f78;
        font-family: 'Manrope', sans-serif;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .bottom-text {
        color: #657681;
        font-size: 14px;
        line-height: 1.8;
    }

    /* ---------------------------------------------
       RESPONSIVE DESIGN
    --------------------------------------------- */
    @media (max-width: 700px) {
        .hero {
            padding: 27px 22px;
        }

        .hero-decoration {
            display: none;
        }

        .stats-grid {
            grid-template-columns: 1fr;
        }

        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .project-description {
            min-height: auto;
        }

        .topbar {
            align-items: flex-start;
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-decoration">🧠</div>
    <div class="hero-tag">WELCOME TO YOUR WORKSPACE</div>
    <h1>Machine Learning<br>Model Hub</h1>
    <p>
        Explore intelligent solutions powered by machine learning.
        Choose a project, configure your input data, and generate
        predictions through a clean and interactive workspace.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# STATISTICS
# --------------------------------------------------
st.markdown("""
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-icon">🧠</div>
        <div class="stat-number">02</div>
        <div class="stat-label">Available ML Projects</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📈</div>
        <div class="stat-number">01</div>
        <div class="stat-label">Regression Model</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🌱</div>
        <div class="stat-number">01</div>
        <div class="stat-label">Classification Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROJECT CARDS
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">Explore Your Projects</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Select a machine learning application to get started.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

# ---------------- MOBILE PRICE PREDICTION ----------------
with col1:
    with st.container(border=True):
        st.markdown("""
        <div class="project-icon">📱</div>
        <div class="project-title">Mobile Price Prediction</div>
        <div class="project-description">
            Estimate mobile phone prices using hardware specifications
            including RAM, internal storage, processor frequency,
            camera quality, battery capacity, and display details.
        </div>
        <div class="project-type">REGRESSION · GRADIENT BOOSTING</div>
        """, unsafe_allow_html=True)

        st.write("")

        st.page_link(
            "pages/1_mobile_price_prediction.py",
            label="Explore Mobile Prediction →",
            icon="📱",
            use_container_width=True
        )

        # Screenshot of the model comparison table
        show_screenshot(
            filename="mobile_results.png",
            caption="Mobile price prediction: model performance comparison",
            download_key="dl_mobile_img",
        )

# ---------------- CROP PREDICTION ----------------
with col2:
    with st.container(border=True):
        st.markdown("""
        <div class="project-icon">🌾</div>
        <div class="project-title">Crop Prediction</div>
        <div class="project-description">
            Find a suitable crop based on soil nutrients,
            temperature, humidity, soil pH, and rainfall
            using your trained Random Forest model.
        </div>
        <div class="project-type">CLASSIFICATION · RANDOM FOREST</div>
        """, unsafe_allow_html=True)

        st.write("")

        st.page_link(
            "pages/2_crop_prediction.py",
            label="Explore Crop Prediction →",
            icon="🌾",
            use_container_width=True
        )

        # Screenshot of the model comparison table
        show_screenshot(
            filename="crop_results.png",
            caption="Crop prediction: model performance comparison",
            download_key="dl_crop_img",
        )
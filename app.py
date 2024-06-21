# Python In-built packages
from pathlib import Path

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Detection Car Lens Zone",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Main page heading
st.title("Detection Car Lens Zone")

# Sidebar
st.sidebar.header("Config")

# Model Options
model_path = Path(settings.DETECTION_MODEL)

confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# session state for youtube video
# session state for polygon drawing
if "current_frame" not in st.session_state:
    st.session_state["current_frame"] = 1
if "sum_frame" not in st.session_state:
    st.session_state["sum_frame"] = 0
# session for check already capture image
if "already_capture" not in st.session_state:
    st.session_state["already_capture"] = False
if "url" not in st.session_state:
    st.session_state["url"] = ""
if "drawline" not in st.session_state:
    st.session_state["drawline"] = False
if "already_draw" not in st.session_state:
    st.session_state["already_draw"] = False

# Selecting Detection Or Segmentation

model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


helper.play_speed_estimation(confidence, model)

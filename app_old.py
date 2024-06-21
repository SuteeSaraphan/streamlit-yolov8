# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
from streamlit_drawable_canvas import st_canvas

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
model_type = st.sidebar.radio("Select Task", ["Detection", "Segmentation"])


confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# session state for youtube video
if "source_temp" not in st.session_state:
    st.session_state["source_temp"] = ""
if "frame" not in st.session_state:
    st.session_state["frame"] = ""
if "current_frame_youtube" not in st.session_state:
    st.session_state["current_frame_youtube"] = 0
# session state for polygon drawing
if "current_frame" not in st.session_state:
    st.session_state["current_frame"] = 1
if "sum_frame" not in st.session_state:
    st.session_state["sum_frame"] = 0
# session for check already capture image
if "already_capture" not in st.session_state:
    st.session_state["already_capture"] = False
if "current_frame_rtsp" not in st.session_state:
    st.session_state["current_frame_rtsp"] = 0
if "web_frame" not in st.session_state:
    st.session_state["web_frame"] = 0
if "rtsp_frame" not in st.session_state:
    st.session_state["rtsp_frame"] = 0

if "url" not in st.session_state:
    st.session_state["url"] = ""
if "interval" not in st.session_state:
    st.session_state["interval"] = 1
if "mxa_detect" not in st.session_state:
    st.session_state["max_detect"] = 0
if "intervaldata" not in st.session_state:
    st.session_state["intervaldata"] = 1
if "drawline" not in st.session_state:
    st.session_state["drawline"] = False
if "already_draw" not in st.session_state:
    st.session_state["already_draw"] = False

# Selecting Detection Or Segmentation

if model_type == "Detection":
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == "Segmentation":
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected

if source_radio == settings.IMAGE:
    # helper.interval_menu('image')
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )
    if source_img is not None:
        uploaded_image = PIL.Image.open(source_img)
        helper.polygon_draw(uploaded_image, source="image")
    else:
        helper.test2 = {}
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(
                    default_image_path, caption="Default Image", use_column_width=True
                )
            else:
                st.image(source_img, caption="Uploaded Image", use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(
                default_detected_image_path,
                caption="Detected Image",
                use_column_width=True,
            )

        else:
            if st.sidebar.button("Detect Objects"):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption="Detected Image", use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)


elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

elif source_radio == settings.SPEED_ESTIMATION:
    helper.play_speed_estimation(confidence, model)

else:
    st.error("Please select a valid source type!")

from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
from count_obj import CountObject
import settings
import torch
import pandas as pd
import PIL
import math
import json
import requests
import settings
from streamlit_drawable_canvas import st_canvas
import speed_check
import base64


result_polygon = []


def polygon_draw(image, width=None, height=None, source=None):
    result_polygon = []

    if source == "image" and type(image) != str:
        scale_x = image.size[0] / 400
        scale_y = image.size[1] / 400
        bg_img = image
    elif source == "video":

        # check frame video
        if st.session_state["sum_frame"] != image.get(cv2.CAP_PROP_FRAME_COUNT):
            st.session_state["sum_frame"] = image.get(cv2.CAP_PROP_FRAME_COUNT)
            st.session_state["current_frame"] = 0

        if st.button("Next Frame"):
            if st.session_state["current_frame"] > image.get(cv2.CAP_PROP_FRAME_COUNT):
                st.session_state["current_frame"] = 0
            else:
                st.session_state["current_frame"] += 30
        image.set(cv2.CAP_PROP_POS_FRAMES, st.session_state["current_frame"])
        ret, frame = image.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        bg_img = frame
        scale_x = frame.size[0] / 400
        scale_y = frame.size[1] / 400
    else:
        return result_polygon

    st.write(
        "Draw a polygon on the image below (Left Click: To Draw a polygon, Right Click: To Enter Polygon):"
    )
    try:

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=1,  # Stroke width
            stroke_color="#ffffff",
            background_image=bg_img,
            height=400,
            width=400,
            drawing_mode="polygon",  # Shape of brush
            key="canvas",
            display_toolbar=True,
            update_streamlit=False if source == "video" else True,
        )
    except Exception as e:
        st.write("Error to draw polygon try again")

    # drawable canvas lib return polygon in json format
    if canvas_result.image_data is not None:
        df = pd.json_normalize(canvas_result.json_data["objects"])
        if len(df) != 0:
            df = df[["path"]]
            # tranform dataframe to list and scale back to normal image
            for i in range(0, len(df["path"])):
                new_df = []
                if type(df["path"][i]) == list:
                    for x in range(0, len(df["path"][i])):
                        if x != len(df["path"][i]) - 1:
                            df["path"][i][x][1] = math.ceil(
                                df["path"][i][x][1] * scale_x
                            )
                            df["path"][i][x][2] = math.ceil(
                                df["path"][i][x][2] * scale_y
                            )
                        new_df.append(df["path"][i][x])
                else:
                    break
                result_polygon.append(new_df)
    return result_polygon


def load_model(model_path):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    return model


def display_tracker_options():
    # display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    display_tracker = "Yes"
    is_display_tracker = True if display_tracker == "Yes" else False
    if is_display_tracker:
        # tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        tracker_type = "bytetrack.yaml"
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(
    conf, model, st_frame, image, is_display_tracking=None, tracker=None
):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(
        res_plotted, caption="Detected Video", channels="BGR", use_column_width=True
    )


def display_count(st_frame, image):
    st_frame.image(
        image, caption="Detected Video", channels="BGR", use_column_width=True
    )


def play_youtube_video(conf, model):
    first_time = False
    # btn
    # source video url
    source_youtube = st.sidebar.text_input("YouTube Video url")
    btn_load = st.sidebar.button("load video")
    interval_menu()
    # check same video
    if st.session_state["source_temp"] != source_youtube:
        first_time = True
        st.session_state["current_frame_youtube"] = 0
        st.session_state["source_temp"] = source_youtube

    is_display_tracker, tracker = display_tracker_options()
    with st.container():
        btn_next = st.button("Next Frame")
        if btn_next and source_youtube != "":
            st.session_state["current_frame_youtube"] += 30
        if st.button("Reset Frame"):
            st.session_state["source_temp"] = ""
            st.session_state["current_frame_youtube"] = 0
            st.session_state["frame"] = 0
            source_youtube = ""

    if source_youtube != "" and first_time or btn_load or btn_next:
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)
            if st.session_state["current_frame_youtube"] < vid_cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            ):
                vid_cap.set(
                    cv2.CAP_PROP_POS_FRAMES, st.session_state["current_frame_youtube"]
                )
            else:
                st.session_state["current_frame_youtube"] = 0
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = vid_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = PIL.Image.fromarray(frame)
            st.session_state["frame"] = frame
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

    if source_youtube != "":
        try:
            result_polygon = polygon_draw(st.session_state["frame"], source="image")

        except Exception as e:
            st.write(e)

    if st.sidebar.button("Detect Objects"):
        print(source_youtube)
        # try:
        yt = YouTube(source_youtube)
        stream = yt.streams.filter(file_extension="mp4", res=720).first()
        detect(stream.url, result_polygon, conf, model)

        # except Exception as e:
        #     st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption(
        "Example URL: rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
    )
    is_display_tracker, tracker = display_tracker_options()
    interval_menu()
    if st.button("Capture"):
        st.session_state["already_capture"] = False
    if st.session_state["already_capture"] == False and source_rtsp != "":
        st.session_state["rtsp_frame"] = 0
        vid_cap = cv2.VideoCapture(source_rtsp)
        if vid_cap.isOpened():
            ret, frame = vid_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        st.session_state["rtsp_frame"] = frame
        st.session_state["already_capture"] = True
    if source_rtsp != "":
        try:
            result_polygon = polygon_draw(
                st.session_state["rtsp_frame"], source="image"
            )
        except Exception as e:
            if e == "object of type 'float' has no len()":
                pass

    if st.sidebar.button("Detect Objects") and source_rtsp != "":
        try:
            detect(source_rtsp, result_polygon, conf, model)
        except Exception as e:
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    interval_menu()
    if st.button("Cupture again"):
        st.session_state["already_capture"] = False
    if st.session_state["already_capture"] == False:
        print("capture")
        vid_cap = cv2.VideoCapture(source_webcam)
        if vid_cap.isOpened():
            ret, frame = vid_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        st.session_state["web_frame"] = frame
        st.session_state["already_capture"] = True
    if st.session_state["already_capture"] == True:
        try:
            result_polygon = polygon_draw(st.session_state["web_frame"], source="image")
        except Exception as e:
            if e == "object of type 'float' has no len()":
                pass
            else:
                st.write("Error to draw polygon try again")

    if st.sidebar.button("Detect Objects"):
        try:
            detect(source_webcam, result_polygon, conf, model)

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    conf: confidence threshold
    scaler: scale image from polygon to scale back normal image
    list_polygon: list of polygon
    example: list_polygon = [[[100,300],[100,100],[1000,100],[1000,300]],[[300,900],[300,300],[1000,300],[1000,900]]]

    """
    interval_menu()

    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())

    if source_vid is not None:
        vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
        result_polygon = polygon_draw(vid_cap, source="video")
        print(result_polygon)
    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), "rb") as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button("Detect Video Objects"):
        try:
            detect(
                str(settings.VIDEOS_DICT.get(source_vid)), result_polygon, conf, model
            )

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def detect_old(source, list_polygon, conf, model):

    vid_cap = cv2.VideoCapture(source)
    st_frame = st.empty()
    btn = st.button("stop")
    # Get video info
    h, w = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    desired_frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / desired_frame_rate
    frames_processed = 0
    frames_to_skip = 0
    start_time = time.time()
    lastest_time = time.time()
    count = CountObject(conf, model, h, w, list_polygon)
    json_display = st.empty()
    jsons = {}
    while vid_cap.isOpened():
        success, image = vid_cap.read()

        if success:
            # Process and display every nth frame, where n is frames_to_skip + 1
            if frames_processed % (frames_to_skip + 1) == 0:
                process_start_time = time.time()

                # Process and display the frame
                img_processed = count.process_frame(image, 0)
                display_count(st_frame, img_processed)
                jsons = json.dumps(count.object_counts)
                json_display.json(jsons)

                process_end_time = time.time()
                processing_time = process_end_time - process_start_time
                frames_to_skip = max(
                    0, math.floor((processing_time - frame_time) / frame_time)
                )
                print(
                    f"Processing time: {processing_time}, Frames to skip: {frames_to_skip}"
                )
                frames_processed = 0

            frames_processed += 1
        else:
            vid_cap.release()
            break
        # if somezone over max car
        if st.session_state["max_detect"] != 0 and st.session_state["url"] != "":
            # send until over st.session_state['intervaldata']
            if any(
                count.object_counts[key] > st.session_state["max_detect"]
                for key in count.object_counts
            ):
                post_api(jsons)
                lastest_time = time.time()
            elif time.time() - lastest_time < st.session_state["intervaldata"]:
                post_api(jsons)

        # if interval time
        elif (
            st.session_state["interval"] != 0
            and time.time() - start_time > st.session_state["interval"]
        ):
            start_time = time.time()
            post_api(jsons)

        if btn:
            st.write("stop")
    if btn:
        post_api(jsons)
        print(response)
        st.write(response.text)


def detect_speed(source, list_line, conf, model):
    vid_cap = cv2.VideoCapture(source)
    w, h, fps = (
        int(vid_cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )
    line_pts = list_line
    st_frame = st.empty()
    btn = st.button("stop")
    # Get video info
    speed_obj = speed_check.SpeedEstimator()
    speed_obj.set_args(reg_pts=line_pts, names=model.model.names)
    desired_frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / desired_frame_rate
    frames_processed = 0
    frames_to_skip = 0
    while vid_cap.isOpened():
        success, image = vid_cap.read()

        if success:
            # Process and display every nth frame, where n is frames_to_skip + 1
            if frames_processed % (frames_to_skip + 1) == 0:
                process_start_time = time.time()
                tracks = model.track(image, persist=True, show=False)

                image = speed_obj.estimate_speed(image, tracks)
                display_count(st_frame, image)
                print(speed_obj.get_speed_data())
                process_end_time = time.time()
                processing_time = process_end_time - process_start_time
                frames_to_skip = max(
                    0, math.floor((processing_time - frame_time) / frame_time)
                )
                print(
                    f"Processing time: {processing_time}, Frames to skip: {frames_to_skip}"
                )
                frames_processed = 0

            frames_processed += 1
        else:
            vid_cap.release()
            break

        if btn:
            st.write("stop")


def convert_to_base64(image):
    _, im_arr = cv2.imencode(
        ".jpg", image
    )  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64


def interval_menu(typeCheck=None):
    option = st.selectbox("Select mode", ("interval", "max_detect"))
    col1, col2, col3 = st.columns(3)
    if typeCheck == "image":
        with col1:
            st.write("input url to send detect data object")
            st.session_state["url"] = st.text_input("url to send data:")
            option = None
    else:
        with col1:
            st.write("input url to send detect data object")
            st.session_state["url"] = st.text_input("url to send data:")
        if option == "interval":
            with col2:
                st.write("interval time to detect object")
                st.session_state["interval"] = st.number_input(
                    "interval time:", value=1
                )
        if option == "max_detect":
            with col2:
                st.write("interval to hold data to send")
                st.session_state["intervaldata"] = st.number_input(
                    "interval time:", value=1
                )
            with col3:
                st.write("max car to detect object")
                st.session_state["max_detect"] = st.number_input("max car", value=0)


def post_api(data):
    if st.session_state["url"] != "":
        try:
            response = requests.post(
                st.session_state["url"],
                data=data,
                headers={"Content-Type": "application/json"},
            )
            print(response)
        except Exception as e:
            pass

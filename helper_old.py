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
from streamlit_drawable_canvas import st_canvas
import speed_check
import base64
import datetime
from json_class import IPCameraDatabase
import os
from urllib.parse import urlparse


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

result_polygon = []
IPcamera_obj = IPCameraDatabase(settings.LINE_PATH)


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


def line_draw(image, width=None, height=None, source=None):

    if source == "image" and type(image) != str:
        scale_x = image.size[0] / 640
        scale_y = image.size[1] / 360
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
        scale_x = frame.size[0] / 640
        scale_y = frame.size[1] / 360

    st.write(
        "Draw a line on the image below (Left Click: To Draw a line, Right Click: To Enter line):"
    )
    try:

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=1,  # Stroke width
            stroke_color="#ffffff",
            background_image=bg_img,
            height=360,
            width=640,
            drawing_mode="line",  # Shape of brush
            key="canvas",
            display_toolbar=True,
            update_streamlit=False if source == "video" else True,
        )
    except Exception as e:
        st.write("Error to draw line try again")

    # print(canvas_result.json_data)
    # drawable canvas lib return polygon in json format
    if canvas_result.image_data is not None:
        df = pd.json_normalize(canvas_result.json_data["objects"])
        # if len(df) != 0:
        #     df = df[["path"]]
        #     # tranform dataframe to list and scale back to normal image
        #     for i in range(0, len(df["path"])):
        #         new_df = []
        #         if type(df["path"][i]) == list:
        #             for x in range(0, len(df["path"][i])):
        #                 if x != len(df["path"][i]) - 1:
        #                     df["path"][i][x][1] = math.ceil(
        #                         df["path"][i][x][1] * scale_x
        #                     )
        #                     df["path"][i][x][2] = math.ceil(
        #                         df["path"][i][x][2] * scale_y
        #                     )
        #                 new_df.append(df["path"][i][x])
        #         else:
        #             break
        #         result_line.append(new_df)

        left = df.loc[0, "left"]
        top = df.loc[0, "top"]
        width = df.loc[0, "width"]
        height = df.loc[0, "height"]

        # Calculate x1, y1
        x1 = left - (width / 2)
        y1 = top - (height / 2)

        # Calculate x2, y2
        x2 = left + (width / 2)
        y2 = top + (height / 2)

        # *400 because the image is resized to 640x360
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y

    return [[x1, y1], [x2, y2]]


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


def play_speed_estimation(conf, model):
    # source_rtsp = st.sidebar.text_input("rtsp stream url:")
    source_rtsp = get_ip_camera()
    result_line = IPcamera_obj.get_data_by_ip_camera(source_rtsp)
    ratio_adjust = st.slider("Ratio to adjust speed size", 0.0, 2.0, 1.0)
    st.write(f"ratio adjust: {ratio_adjust}")
    if result_line == None:
        st.write("Please draw line to detect car speed")
        st.session_state["drawline"] = True
    else:
        st.session_state["drawline"] = False
        st.session_state["already_draw"] = True

    if st.button("Reset Line"):
        st.session_state["drawline"] = True
        IPcamera_obj.delete_ip_camera(source_rtsp)

    # is_display_tracker, tracker = display_tracker_options()
    if st.button("Capture"):
        st.session_state["already_capture"] = False
    if st.session_state["already_capture"] == False and source_rtsp != "":
        st.session_state["rtsp_frame"] = 0
        print(source_rtsp)
        vid_cap = cv2.VideoCapture(source_rtsp)
        if vid_cap.isOpened():
            ret, frame = vid_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        st.session_state["rtsp_frame"] = frame
        st.session_state["already_capture"] = True
    if source_rtsp != "" and st.session_state["drawline"]:
        try:
            print("draw line")
            result_line = line_draw(st.session_state["rtsp_frame"], source="image")
            if IPcamera_obj.edit_ip_camera(source_rtsp, result_line):
                st.write("Draw line success")
            else:
                IPcamera_obj.add_ip_camera(source_rtsp, result_line)

            print(f"Already draw: {st.session_state['already_draw']}")
            print(f"Result line: {result_line}")
            st.session_state["already_draw"] = True
            st.session_state["drawline"] = False

            # IPcamera_obj.set_data_by_ip_camera(source_rtsp, result_line)
        except Exception as e:
            if e == "object of type 'float' has no len()":
                pass
            else:
                print(e)

    # result_polygon = IPcamera_obj.get_data_by_ip_camera(source_rtsp)
    # st.write(f"IP Camera: {source_rtsp} list line: {result_polygon}")
    # if st.button("Run Speed Estimation") and source_rtsp != "":
    #     try:
    #         detect_speed(source_rtsp != "", result_polygon, conf, model)
    #     except Exception as e:
    #         st.sidebar.error("Error loading RTSP stream: " + str(e))

    # test
    st.write(f"IP Camera: {source_rtsp} list line: {result_line}")
    if st.button("Run Speed Estimation") and source_rtsp != "" and result_line != None:
        try:
            detect_speed(
                "videos/test_real_2.mp4",
                result_line,
                conf,
                model,
                source_rtsp,
                ratio_adjust,
            )
        except Exception as e:
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def detect_speed(source, list_line, conf, model, source_rtsp=None, ratio_adjust=1.0):
    st_frame = st.empty()
    btn = st.button("stop")
    vid_cap = cv2.VideoCapture(source)
    w, h, fps = (
        int(vid_cap.get(prop))
        for prop in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
        )
    )

    speed_obj = speed_check.SpeedEstimator()
    speed_obj.set_args(reg_pts=list_line, names=model.model.names, view_img=False)

    frame_time = 1.0 / fps
    frames_processed = 0
    frames_to_skip = 0
    prev_Frame = 0
    list_check = []

    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if not success:
            break

        # if frames_processed % (frames_to_skip + 1) == 0:
        #     current_Frame = vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
        #     frame_diff = int(current_Frame - prev_Frame)
        #     process_start_time = time.time()

        #     tracks = model.track(
        #         image,
        #         persist=True,
        #         imgsz=(h / 4, w / 4),
        #         classes=[2, 5, 7],
        #         vid_stride=frame_diff,
        #         conf=conf,
        #     )

        #     time_to_skip = frame_diff * frame_time
        #     image_speed = speed_obj.estimate_speed(image, tracks, time_to_skip)
        #     display_count(st_frame, image_speed)

        #     dict_data = process_speed_data(
        #         speed_obj, list_check, image, source_rtsp, ratio_adjust
        #     )
        #     if dict_data:
        #         response = post_api(json.dumps(dict_data))
        #         print(response)
        #     processing_time = time.time() - process_start_time
        #     frames_to_skip = max(
        #         0, math.floor((processing_time - frame_time) / frame_time)
        #     )

        #     frames_processed = 0
        #     prev_Frame = current_Frame

        # frames_processed += 1

        tracks = model.track(
            image,
            persist=True,
            imgsz=(h / 4, w / 4),
            classes=[2, 5, 7],
            conf=conf,
        )

        image_speed = speed_obj.estimate_speed(image, tracks)
        display_count(st_frame, image_speed)

        dict_data = process_speed_data(
            speed_obj, list_check, image, source_rtsp, ratio_adjust
        )
        if dict_data:
            post_api(json.dumps(dict_data))

        if btn:
            st.write("stop")
            break

    vid_cap.release()


def process_speed_data(speed_obj, list_check, image, source_rtsp, raito_adjust=1.0):
    dict_data = speed_obj.get_speed_data()
    if dict_data == {}:
        return None
    for key, value in dict_data.items():
        if key in list_check:
            speed_obj.clear_dict_data(key)
            continue
        list_check.append(key)

        h, w = image.shape[:2]
        print(f"image shape: {h}x{w}")
        x_ratio = int(w * 0.05)
        y_ratio = int(h * 0.05)
        print(f"x_ratio: {x_ratio}, y_ratio: {y_ratio}")
        x1, y1, x2, y2 = (
            value[1][0] - x_ratio,
            value[1][1] - y_ratio,
            value[1][2] + x_ratio,
            value[1][3] + y_ratio,
        )
        while x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            x_ratio = int(x_ratio * 0.5)
            y_ratio = int(y_ratio * 0.5)
            x1, y1, x2, y2 = (
                value[1][0] - x_ratio,
                value[1][1] - y_ratio,
                value[1][2] + x_ratio,
                value[1][3] + y_ratio,
            )
            print(f"Update x_ratio: {x_ratio}, y_ratio: {y_ratio}")
        crop_img = image[y1:y2, x1:x2]
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
        imgbase64_full = convert_to_base64(image)
        imgbase64_crop = convert_to_base64(crop_img)
        length = float(x2 - x1) * raito_adjust
        speed = value[0] * raito_adjust
        vehicle_dict = create_dict(
            urlparse(source_rtsp).netloc,
            key,
            length,
            speed,
            imgbase64_full,
            imgbase64_crop,
        )
        return vehicle_dict


def convert_to_base64(image):
    _, im_arr = cv2.imencode(
        ".jpg", image
    )  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    im_b64 = im_b64.decode("utf-8")
    return im_b64


def get_ip_camera():
    col1, col2 = st.columns(2)
    with col1:
        with st.form("my-url-form"):
            st.write("input url to send detect data object Ex : http://127.0.0.1:5000")
            st.session_state["url"] = st.text_input("url to send data:")
            submit_button = st.form_submit_button("Submit URL")
    with col2:
        with st.form("my-RTSP"):
            st.write("Example URL: rtsp://localhost:8554/")
            ip_rtsp = st.text_input("Input IP RTSP :")
            submit_button = st.form_submit_button("Submit IP RTSP")
    return ip_rtsp


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
            return response
        except Exception as e:
            pass


def create_dict(
    ip_camera,
    id,
    length,
    speed,
    imgbase64_full,
    imgbase64_crop,
    license_province=None,
    license_front=None,
    license_back=None,
):
    """
    Create a dictionary containing information about a vehicle.

    Parameters:
        ip_camera (str): The IP address of the camera capturing the vehicle.
        length (float): The length of the vehicle.
        speed (float): The speed of the vehicle.
        imgbase64 (str): The base64 encoded image of the vehicle.
        license_province (str, optional): The province of the vehicle's license plate.
        license_front (str, optional): The front image of the vehicle's license plate.
        license_back (str, optional): The back image of the vehicle's license plate.

    Returns:
        dict: A dictionary containing the vehicle information.
    """
    # Type checking
    if not isinstance(ip_camera, str):
        raise TypeError("ip_camera must be a string")
    if not isinstance(length, float):
        raise TypeError("length must be a float")
    if not isinstance(id, int):
        raise TypeError("id must be an integer")
    if not isinstance(speed, float):
        raise TypeError("speed must be a float")
    if not isinstance(imgbase64_full, str):
        raise TypeError("imgbase64 must be a string")
    if not isinstance(imgbase64_crop, str):
        raise TypeError("imgbase64 must be a string")
    if license_province is not None and not isinstance(license_province, str):
        raise TypeError("license_province must be a string or None")
    if license_front is not None and not isinstance(license_front, str):
        raise TypeError("license_front must be a string or None")
    if license_back is not None and not isinstance(license_back, str):
        raise TypeError("license_back must be a string or None")

    # Create the dictionary
    vehicle_dict = {}
    vehicle_dict["cameraIp"] = ip_camera
    vehicle_dict["trxDatetime"] = str(datetime.datetime.now())
    vehicle_dict["vehicleId"] = id
    vehicle_dict["vehicleLength"] = length
    vehicle_dict["vehicleSpeed"] = speed
    vehicle_dict["licencePlateProvince"] = license_province
    vehicle_dict["licencePlateFront"] = license_front
    vehicle_dict["licencePlateBack"] = license_back
    vehicle_dict["vehicleImage_full"] = imgbase64_full
    vehicle_dict["vehicleImage_crop"] = imgbase64_crop
    return vehicle_dict


if __name__ == "__main__":
    model = load_model("yolov8s.pt")
    detect_speed("videos/video1.mp4", [[0, 360], [1920, 360]], 0.5, model)

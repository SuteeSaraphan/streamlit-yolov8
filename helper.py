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
import settings
from streamlit_drawable_canvas import st_canvas



result_polygon = []


def polygon_draw(image, width = None, height = None , source = None):
    if source == "image":
        result_polygon = []
        scale_x = image.size[0] / 400
        scale_y = image.size[1] / 400
        bg_img = image
    elif source == "video":
        result_polygon = []
        if st.button("Next Frame"):
            st.session_state['current_frame'] += 30
        image.set(cv2.CAP_PROP_POS_FRAMES, st.session_state['current_frame'])
        ret , frame = image.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        bg_img = frame
        scale_x = frame.size[0] / 400
        scale_y = frame.size[1] / 400

    st.write("Draw a polygon on the image below (Left Click: To Draw a polygon, Right Click: To Enter Polygon):")
    canvas_result = st_canvas(
                fill_color='rgba(255, 165, 0, 0.3)',  # Fixed fill color with some opacity
                stroke_width=1,
                stroke_color='#ffffff',
                background_image=bg_img,
                height=400,
                width=400,
                drawing_mode='polygon',
                key='canvas',
                display_toolbar=True,
                update_streamlit=False if source == "video" else True,
            )
    if canvas_result.image_data is not None:
            df = pd.json_normalize(canvas_result.json_data["objects"])
            if (len(df) != 0):
                new_df = []
                df = df[['path']]
                for i in range(0, len(df['path'])):
                    new_df = []
                    for x in range(0, len(df['path'][i])):
                        if x != len(df['path'][i]) - 1:
                            df['path'][i][x][1] = math.ceil(df['path'][i][x][1] * scale_x)
                            df['path'][i][x][2] = math.ceil(df['path'][i][x][2] * scale_y)
                        new_df.append(df['path'][i][x])
                    result_polygon.append(new_df)
                    print(result_polygon)
    return result_polygon

def load_model(model_path):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    print(device)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def display_count(st_frame, image):
    st_frame.image(image,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    print(source_youtube)
    is_display_tracker, tracker = display_tracker_options()
    if source_youtube != "":
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)
            result_polygon=polygon_draw(vid_cap, source = "video")
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
       
    if st.sidebar.button('Detect Objects'):
        try:
            detect(stream.url,result_polygon,conf,model)
            # st_frame = st.empty()
            # while (vid_cap.isOpened()):
            #     success, image = vid_cap.read()
            #     if success:
            #         _display_detected_frames(conf,
            #                                  model,
            #                                  st_frame,
            #                                  image,
            #                                  is_display_tracker,
            #                                  tracker,
            #                                  )
            #     else:
            #         vid_cap.release()
            #         break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if source_rtsp != "":
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            result_polygon=polygon_draw(vid_cap, source = "video")
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
       
    if st.sidebar.button('Detect Objects'):
        try:
            detect(source_rtsp,result_polygon,conf,model)
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    capture = True
    vid_cap = cv2.VideoCapture(source_webcam)
    if st.button("capture"):
        capture = False
    if (vid_cap.isOpened()):
        ret , frame = vid_cap.read()
            
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = PIL.Image.fromarray(frame)
    result_polygon=polygon_draw(frame, source = "image")
   
    if st.sidebar.button('Detect Objects'):
        try:
            detect(source_webcam,result_polygon,conf,model)
            # vid_cap = cv2.VideoCapture(source_webcam)
            # st_frame = st.empty()
            # while (vid_cap.isOpened()):
            #     success, image = vid_cap.read()
            #     if success:
            #         _display_detected_frames(conf,
            #                                  model,
            #                                  st_frame,
            #                                  image,
            #                                  is_display_tracker,
            #                                  tracker,
            #                                  )
            #     else:
            #         vid_cap.release()
            #         break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf,model):
    '''
        conf: confidence threshold
        scaler: scale image from polygon to scale back normal image
        list_polygon: list of polygon
        example: list_polygon = [[[100,300],[100,100],[1000,100],[1000,300]],[[300,900],[300,300],[1000,300],[1000,900]]]
    
    '''    
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    if source_vid is not None:
        vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
        result_polygon = polygon_draw(vid_cap, source = "video")
    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
 
    if st.sidebar.button('Detect Video Objects'):
        try:
            detect(str(settings.VIDEOS_DICT.get(source_vid)),result_polygon,conf,model)
            # vid_cap = cv2.VideoCapture(
            #     str(settings.VIDEOS_DICT.get(source_vid)))
            # st_frame = st.empty()
            # #get video info
            # h,w = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            
            # count = CountObject(conf,model,h,w,list_polygon)

            # text=st.empty()
            # while (vid_cap.isOpened()):
            #     success, image = vid_cap.read()
            #     if success:
            #         img_processed,json = count.process_frame(image, 0)
            #         # _display_detected_frames(conf,
            #         #                          model,
            #         #                          st_frame,
            #         #                          image,
            #         #                          is_display_tracker,
            #         #                          tracker
            #         #                          )
            #         display_count(st_frame,img_processed)
            #         text.json(json)
            #     else:
            #         vid_cap.release()
            #         break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def detect(source, list_polygon, conf, model):
    vid_cap = cv2.VideoCapture(source)
    st_frame = st.empty()
    
    # Get video info
    h, w = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    count = CountObject(conf, model, h, w, list_polygon)
    text = st.empty()
    
    frame_counter = 0
    
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        frame_counter += 1
        
        if success:
            if frame_counter % 25 == 0:
                img_processed, json = count.process_frame(image, 0)
                display_count(st_frame, img_processed)
                text.json(json)
                frame_counter = 0
    
        else:
            vid_cap.release()
            break

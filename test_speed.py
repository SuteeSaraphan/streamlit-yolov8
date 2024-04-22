from ultralytics import YOLO
import speed_check
import cv2
import time
import math

model = YOLO("yolov8s.pt")
names = model.model.names

vid_cap = cv2.VideoCapture("videos/video1.mp4")
assert vid_cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(vid_cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Video writer

line_pts = [(0, 360), (1280, 360)]

# Init speed-estimation obj
speed_obj = speed_check.SpeedEstimator()
speed_obj.set_args(reg_pts=line_pts, names=names)
desired_frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / desired_frame_rate
frames_processed = 0
frames_to_skip = 0
start_time = time.time()
lastest_time = time.time()
while vid_cap.isOpened():
    success, image = vid_cap.read()

    if success:
        # Process and display every nth frame, where n is frames_to_skip + 1
        if frames_processed % (frames_to_skip + 1) == 0:
            process_start_time = time.time()
            tracks = model.track(image, persist=True, show=False)

            image = speed_obj.estimate_speed(image, tracks)
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

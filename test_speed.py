from ultralytics import YOLO
import speed_check
import cv2
import time
import math
import torch
import os
import datetime
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def detect_speed(source, list_line, conf, model):
    vid_cap = cv2.VideoCapture(source)
    w, h, fps = (
        int(vid_cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )
    line_pts = list_line
    # Get video info
    speed_obj = speed_check.SpeedEstimator()
    speed_obj.set_args(reg_pts=line_pts, names=model.model.names, view_img=False)
    desired_frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
    print(f"Desired frame rate: {desired_frame_rate}")
    frame_time = 1.0 / desired_frame_rate
    print(f"Frame time: {frame_time}")
    print(frame_time * desired_frame_rate)
    frames_processed = 0
    frames_to_skip = 0
    prev_Frame = 0
    list_check = []
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            # Process and display every nth frame, where n is frames_to_skip + 1
            if frames_processed % (frames_to_skip + 1) == 0:
                current_Frame = vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
                frame_diff = int(current_Frame - prev_Frame)
                process_start_time = time.time()
                tracks = model.track(
                    image,
                    persist=True,
                    imgsz=(h, w),
                    classes=[2, 5, 7],
                    vid_stride=frame_diff,
                    conf=conf,
                )
                time_to_skip = frame_diff * frame_time
                image_speed = speed_obj.estimate_speed(image, tracks, time_to_skip)
                # display_count(st_frame, image)

                dict_data = speed_obj.get_speed_data()
                # print all keys and values
                for key, value in dict_data.items():
                    print(f"Key: {key}, Value: {value}")
                    if key in list_check:
                        speed_obj.clear_dict_data(key)
                        continue
                    list_check.append(key)
                    # crop image
                    x1, y1, x2, y2 = (
                        value[1][0] - 10,
                        value[1][1] - 10,
                        value[1][2] + 10,
                        value[1][3] + 10,
                    )
                    crop_img = image[y1:y2, x1:x2]
                    cv2.imwrite(f"test/crop_{key}.jpg", crop_img)
                    imgbase64 = convert_to_base64(crop_img)
                    length = float(x2 - x1)
                    speed = value[0]
                    vehicle_dict = create_dict("1.1.1.1", key, length, speed, imgbase64)
                    print(vehicle_dict)

                process_end_time = time.time()
                processing_time = process_end_time - process_start_time
                frames_to_skip = max(
                    0, math.floor((processing_time - frame_time) / frame_time)
                )
                print(
                    f"Processing time: {processing_time}, Frames to skip: {frames_to_skip}"
                )
                frames_processed = 0
                prev_Frame = current_Frame

            frames_processed += 1
        else:
            vid_cap.release()
            break

        # if btn:
        #     st.write("stop")


def convert_to_base64(image):
    _, im_arr = cv2.imencode(
        ".jpg", image
    )  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    im_b64 = im_b64.decode("utf-8")
    return im_b64


def create_dict(
    ip_camera,
    id,
    length,
    speed,
    imgbase64,
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
    if not isinstance(imgbase64, str):
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
    vehicle_dict["vehicleImage"] = imgbase64
    return vehicle_dict


if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)
    detect_speed("videos/test_real_2.mp4", [(0, 300), (640, 300)], 0.4, model)
    # img = cv2.imread("test.jpg")
    # imgbase64 = convert_to_base64(img)
    # vehicle_dict = create_dict("

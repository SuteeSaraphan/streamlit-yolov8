from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
draw_polygon = False
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"
RTSP = "RTSP"
YOUTUBE = "YouTube"
SPEED_ESTIMATION = "Speed Estimation"

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE, SPEED_ESTIMATION]

# Images config
IMAGES_DIR = ROOT / "images"
DEFAULT_IMAGE = IMAGES_DIR / "p1.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "traffic_processed.jpg"

# Videos config
VIDEO_DIR = ROOT / "videos"
VIDEO_1_PATH = VIDEO_DIR / "video1.mp4"
VIDEO_2_PATH = VIDEO_DIR / "video_2.mp4"
VIDEO_3_PATH = VIDEO_DIR / "video_3.mp4"
VIDEO_4_PATH = VIDEO_DIR / "video_1.mp4"
VIDEOS_DICT = {
    "video_1": VIDEO_1_PATH,
    "video_2": VIDEO_2_PATH,
    "video_3": VIDEO_3_PATH,
    "video1": VIDEO_4_PATH,
}

# ML Model config
MODEL_DIR = ROOT / "weights"
DETECTION_MODEL = MODEL_DIR / "yolov8s.pt"
SEGMENTATION_MODEL = MODEL_DIR / "yolov8n-seg.pt"

# line config
LINE_PATH = ROOT / "list_line.json"

# Webcam
WEBCAM_PATH = 0

# Polygon TEST

MOCK_LIST_POLYGON = [
    [["M", 219, 175], ["L", 410, 179], ["L", 347, 314], ["L", 230, 314], ["z"]],
    [
        ["M", 243, 37],
        ["L", 78, 61],
        ["L", 77, 210],
        ["L", 220, 134],
        ["L", 359, 124],
        ["z"],
    ],
]

# Class in yolo
CLASS_SELECT = [2, 5, 7]

from count_obj import CountObject
import cv2
from ultralytics import YOLO
import torch

device: str = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO('yolov8s.pt')
model.to('cuda')
#print(model.model.names)
# Load the array
list1=[[500,800],[500,500],[1200,500],[1200,800]]
list2=[[200,200],[1000,200],[1000,800]]
list = [list1,list2]
# print(array)
img = cv2.imread('images/p1.jpg')
count = CountObject(model, img.shape[0], img.shape[1], list)
img_processed,json = count.process_frame(img, 0)
print(json)
cv2.imwrite('images/traffic_processed.jpg', img_processed)

# import numpy as np
# list1=[[500,800],[500,500],[1200,500],[1200,800]]
# list2=[[200,200],[1000,200],[1000,800]]
# list = [list1,list2]
# scale=2
# print(([np.array(l)*scale for l in list]))
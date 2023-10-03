# from ultralytics import YOLO
# import cv2
import numpy as np
import supervision as sv
import json
class CountObject():
    def __init__(self,confident,model,h,w,list_polygon)-> None:
        self.model = model
        self.colors = sv.ColorPalette.default()
        self.confident = confident
        self.polygons = [np.array([index[1:] for index in l][:-1]).astype(np.int32) for l in list_polygon]
        print(self.polygons)
        
        self.img_height = h
        self.img_width = w
        self.zones = [sv.PolygonZone(polygon=polygon,frame_resolution_wh=(self.img_width, self.img_height)) for polygon in self.polygons]
        
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=self.colors.by_idx(index),
                thickness=3,
                text_thickness=2,
                text_scale=1
            )
            for index, zone
            in enumerate(self.zones)
        ]
        
        self.box_annotators = [
            sv.BoxAnnotator(
                color=self.colors.by_idx(index),
                thickness=4,
                text_thickness=4,
                text_scale=2
                )
            for index
            in range(len(self.polygons))
        ]

        # Initialize a dictionary to store object counts for each polygon zone
        self.object_counts = {index: 0 for index in range(len(self.polygons))}
        

    def process_frame(self, frame: np.ndarray, i) -> np.ndarray:
        dict_data={}
        # detect
        results = self.model(frame, imgsz=self.img_width)[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = detections[(detections.class_id == 2) & (detections.confidence > self.confident)]

        for index, (zone, zone_annotator, box_annotator) in enumerate(zip(self.zones, self.zone_annotators, self.box_annotators)):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]

            # Update object count for this zone
            self.object_counts[index] = len(detections_filtered)


            # Draw the count as text on the frame
            # count_text = f'Count: {self.object_counts[index]}'
            # frame = cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
            frame = zone_annotator.annotate(scene=frame)

          # Print object counts for each zone
        for index, count in self.object_counts.items():
            print(f'Zone{index + 1}: {count}')
            #create json file
            data = {}
            data[f'zone{index + 1}'] = count
            dict_data.update(data)

        #Change dict_data to json file
        json_data = json.dumps(dict_data, indent=4)
        

        return frame,json_data
    
        
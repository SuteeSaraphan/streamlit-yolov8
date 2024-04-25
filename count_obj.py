# from ultralytics import YOLO
# import cv2
import numpy as np
import supervision as sv
import json
import settings
import streamlit as st


class CountObject:
    def __init__(self, confident, model, h, w, list_polygon) -> None:
        self.model = model
        self.byte_tracker = sv.ByteTrack()
        self.colors = sv.ColorPalette.default()
        self.confident = confident
        self.polygons = [
            np.array([index[1:] for index in l][:-1]).astype(np.int32)
            for l in list_polygon
        ]
        print(self.polygons)

        self.img_height = h
        self.img_width = w
        self.zones = [
            sv.PolygonZone(
                polygon=polygon, frame_resolution_wh=(self.img_width, self.img_height)
            )
            for polygon in self.polygons
        ]

        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=self.colors.by_idx(index),
                thickness=3,
                text_thickness=2,
                text_scale=1,
            )
            for index, zone in enumerate(self.zones)
        ]

        self.box_annotators = [
            sv.BoxAnnotator(
                color=self.colors.by_idx(index),
                thickness=4,
                text_thickness=4,
                text_scale=2,
            )
            for index in range(len(self.polygons))
        ]

        # Initialize a dictionary to store object counts for each polygon zone
        self.object_counts = {index: 0 for index in range(len(self.polygons))}
        self.object_tracks = {}

    def process_frame(self, frame: np.ndarray, i) -> np.ndarray:
        # dict_data={}
        # detect
        results = self.model(frame, imgsz=self.img_width)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[
            np.isin(detections.class_id, [2, 5, 7])
            & (detections.confidence > self.confident)
        ]
        detections = self.byte_tracker.update_with_detections(detections)

        for index, (zone, zone_annotator, box_annotator) in enumerate(
            zip(self.zones, self.zone_annotators, self.box_annotators)
        ):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            # Update object count for this zone
            self.object_counts[index] = len(detections_filtered)
            for xyxy, mask, confidence, class_id, tracker_id in detections:
                ## want to speed estimation

                if tracker_id not in self.object_tracks:
                    self.object_tracks[tracker_id] = []
                self.object_tracks[tracker_id].append(xyxy)
                # print(self.object_tracks)
                # print(xyxy)
                # print(tracker_id)
                # print(self.object_tracks[tracker_id])
                # print(self.object_tracks[tracker_id][0])

            frame = box_annotator.annotate(
                scene=frame, detections=detections_filtered, skip_label=True
            )
            frame = zone_annotator.annotate(scene=frame)

        return frame

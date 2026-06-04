import os
import cv2
import json

FRAME_DIR = r"C:\Storage\maritime_final\data\frames"

DETECTION_JSON = r"C:\Storage\maritime_final\data\detections\detections.json"

OUTPUT_DIR = r"C:\Storage\maritime_final\data\crops"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DETECTION_JSON, "r") as f:
    detections = json.load(f)

for frame_name, boxes in detections.items():

    frame_path = os.path.join(FRAME_DIR, frame_name)

    frame = cv2.imread(frame_path)

    for idx, det in enumerate(boxes):

        x1, y1, x2, y2 = det["bbox"]

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        save_name = f"{frame_name[:-4]}_{idx}.jpg"

        save_path = os.path.join(OUTPUT_DIR, save_name)

        cv2.imwrite(save_path, crop)

        print(save_path)

print("DONE")
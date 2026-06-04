import os
import cv2
import json

from ultralytics import RTDETR

MODEL_PATH =r"C:\Storage\maritime_v7\runs\detect\runs\coastline_rtdetr-4\weights\best.pt"


FRAME_DIR = r"C:\Storage\maritime_final\data\frames"

OUTPUT_JSON = r"C:\Storage\maritime_final\data\detections\detections.json"

os.makedirs("data/detections", exist_ok=True)

model = RTDETR(MODEL_PATH)

all_detections = {}

frame_files = sorted(os.listdir(FRAME_DIR))

for file in frame_files:

    path = os.path.join(FRAME_DIR, file)

    frame = cv2.imread(path)

    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "bbox": [x1, y1, x2, y2]
            })

    all_detections[file] = detections

    print(file, len(detections))

with open(OUTPUT_JSON, "w") as f:
    json.dump(all_detections, f, indent=4)

print("DONE")
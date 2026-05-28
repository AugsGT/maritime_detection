import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from ultralytics import RTDETR
from sklearn.metrics.pairwise import cosine_similarity

# PATHS

VIDEO_PATH = r"C:\Storage\maritime_final\data\fix2_v.mp4"

REFERENCE_DIR = r"C:\Storage\maritime_final\refer\lighthouse"

MODEL_PATH = r"C:\Storage\maritime_v7\runs\detect\runs\coastline_rtdetr-4\weights\best.pt"

OUTPUT_VIDEO = r"C:\Storage\maritime_final\output\output3.mp4"

DEBUG_DIR = r"C:\Storage\maritime_final\debug"

# CREATE DIRECTORIES

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

SIMILARITY_THRESHOLD = 0.38

# DEVICE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)

# LOAD MODELS

print("Loading RT-DETR...")
detector = RTDETR(MODEL_PATH)

print("Loading DINOv2...")

dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
)

dinov2.eval()
dinov2.to(DEVICE)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# IMAGE ENHANCEMENT

def enhance_crop(crop):

    # UPSCALE
    crop = cv2.resize(
        crop,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # CLAHE CONTRAST ENHANCEMENT

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    crop = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return crop

# EMBEDDING EXTRACTION

def extract_embedding(image):

    image = enhance_crop(image)

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil = Image.fromarray(rgb)

    tensor = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = dinov2(tensor)

    emb = emb.cpu().numpy().flatten()

    emb = emb / np.linalg.norm(emb)

    return emb

# LOAD REFERENCE EMBEDDINGS

reference_embeddings = []

print("\nLOADING REFERENCES...\n")

for file in os.listdir(REFERENCE_DIR):

    path = os.path.join(REFERENCE_DIR, file)

    image = cv2.imread(path)

    if image is None:
        continue

    emb = extract_embedding(image)

    reference_embeddings.append(emb)

    print("REFERENCE:", file)

reference_embeddings = np.array(reference_embeddings)

print("\nTOTAL REFERENCES:", len(reference_embeddings))
==============================================

tracks = {}

next_track_id = 0

# IOU FUNCTION

def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)

    interArea = interW * interH

    areaA = (
        (boxA[2] - boxA[0]) *
        (boxA[3] - boxA[1])
    )

    areaB = (
        (boxB[2] - boxB[0]) *
        (boxB[3] - boxB[1])
    )

    union = areaA + areaB - interArea

    if union == 0:
        return 0

    return interArea / union

# VIDEO SETUP

print("\nVIDEO EXISTS:", os.path.exists(VIDEO_PATH))

cap = cv2.VideoCapture(VIDEO_PATH)

print("VIDEO OPEN:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("WIDTH:", width)
print("HEIGHT:", height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

print("WRITER OPEN:", out.isOpened())

# MAIN LOOP


frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    print("\nFRAME:", frame_id)

    shoreline_limit = int(height * 0.85)

    results = detector.predict(
        frame,
        conf=0.15,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:

        print("NO DETECTIONS")

        out.write(frame)

        continue

    print("DETECTIONS:", len(result.boxes))

    # LOOP THROUGH DETECTIONS

    for idx, box in enumerate(result.boxes):

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        w = x2 - x1
        h = y2 - y1

        # BASIC FILTERS

        if w < 30 or h < 30:
            continue

        if y2 > shoreline_limit:
            continue

        aspect_ratio = w / max(h, 1)

        if aspect_ratio < 0.5:
            continue

        if aspect_ratio > 5:
            continue

        # PAD BOX

        pad = 10

        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)

        x2p = min(width, x2 + pad)
        y2p = min(height, y2 + pad)

        crop = frame[
            y1p:y2p,
            x1p:x2p
        ]

        if crop.size == 0:
            continue

        # SAVE DEBUG CROP

        debug_path = os.path.join(
            DEBUG_DIR,
            f"f{frame_id}_d{idx}.jpg"
        )

        cv2.imwrite(debug_path, crop)

        # EXTRACT EMBEDDING
    

        emb = extract_embedding(crop)

        sims = cosine_similarity(
            [emb],
            reference_embeddings
        )[0]

        
        # TOP-K AVERAGE SIMILARITY

        top_k = sorted(
            sims,
            reverse=True
        )[:3]

        similarity = float(np.mean(top_k))

        print(
            f"DET {idx} "
            f"SIM {similarity:.4f}"
        )

        # TRACK MATCHING

        matched_track = None

        for tid, tdata in tracks.items():

            prev_box = tdata["bbox"]

            iou = compute_iou(
                [x1, y1, x2, y2],
                prev_box
            )

            if iou > 0.3:

                matched_track = tid
                break

        if matched_track is None:

            matched_track = next_track_id

            next_track_id += 1

            tracks[matched_track] = {
                "scores": [],
                "bbox": [x1, y1, x2, y2]
            }

        tracks[matched_track]["bbox"] = \
            [x1, y1, x2, y2]

        tracks[matched_track]["scores"].append(
            similarity
        )

        tracks[matched_track]["scores"] = \
            tracks[matched_track]["scores"][-20:]

        avg_similarity = float(np.mean(
            tracks[matched_track]["scores"]
        ))

        print(
            f"TRACK {matched_track} "
            f"AVG {avg_similarity:.4f}"
        )

        # FINAL CLASSIFICATION

        if avg_similarity > SIMILARITY_THRESHOLD:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            label = (
                f"Detected "
                f"{avg_similarity:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

    out.write(frame)

    cv2.imshow("OUTPUT", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()

out.release()

cv2.destroyAllWindows()

print("\nDONE")
print("OUTPUT:", OUTPUT_VIDEO)
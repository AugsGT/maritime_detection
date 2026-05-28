# ============================================================
# PORT OFFICE IDENTIFIER - VIDEO VERSION
# ============================================================
#
# PIPELINE:
#
# VIDEO
#   ↓
# RT-DETR BUILDING DETECTION
#   ↓
# BUILDING CROPS
#   ↓
# DINOv2 EMBEDDINGS
#   ↓
# REFERENCE COMPARISON
#   ↓
# PORT OFFICE IDENTIFICATION
#   ↓
# DRAW BOUNDING BOX
#
# ============================================================

import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from ultralytics import RTDETR
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = r"C:\Storage\maritime_final\data\fixed1.mp4"

OUTPUT_PATH = r"C:\Storage\maritime_final\output\output_detected11.mp4"

REFERENCE_FOLDER = r"C:\Storage\maritime_final\refer"

MODEL_PATH = r"C:\Storage\maritime_v7\runs\detect\runs\coastline_rtdetr-4\weights\best.pt"

SIMILARITY_THRESHOLD = 0.72

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD RT-DETR
# ============================================================

print("[INFO] Loading RT-DETR...")

detector = RTDETR(MODEL_PATH)

print("[INFO] RT-DETR loaded.")

# ============================================================
# LOAD DINOv2
# ============================================================

print("[INFO] Loading DINOv2...")

dinov2 = torch.hub.load(
    'facebookresearch/dinov2',
    'dinov2_vitb14'
)

dinov2.eval()
dinov2.to(DEVICE)

print("[INFO] DINOv2 loaded.")

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# EXTRACT DINO EMBEDDING
# ============================================================

def extract_embedding(image_bgr):

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(image_rgb)

    tensor = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = dinov2(tensor)

    embedding = embedding.cpu().numpy().flatten()

    # normalize vector
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

# ============================================================
# LOAD REFERENCE EMBEDDINGS
# ============================================================

print("\n[INFO] Loading reference images...\n")

reference_embeddings = []

for file in os.listdir(REFERENCE_FOLDER):

    path = os.path.join(REFERENCE_FOLDER, file)

    image = cv2.imread(path)

    if image is None:
        continue

    emb = extract_embedding(image)

    reference_embeddings.append(emb)

    print(f"[REFERENCE] Loaded: {file}")

reference_embeddings = np.array(reference_embeddings)

print(f"\n[INFO] Total references: {len(reference_embeddings)}")

# ============================================================
# VIDEO SETUP
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception("Could not open video.")

fps = int(cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0

# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    print(f"[FRAME] {frame_count}")

    # --------------------------------------------------------
    # RT-DETR DETECTION
    # --------------------------------------------------------

    results = detector.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # LOOP THROUGH DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            # ------------------------------------------------
            # SAFETY CHECKS
            # ------------------------------------------------

            w = x2 - x1
            h = y2 - y1

            if w < 60 or h < 60:
                continue

            # ------------------------------------------------
            # CROP BUILDING
            # ------------------------------------------------

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # ------------------------------------------------
            # DINO EMBEDDING
            # ------------------------------------------------

            candidate_embedding = extract_embedding(crop)

            # ------------------------------------------------
            # SIMILARITY
            # ------------------------------------------------

            similarities = cosine_similarity(
                [candidate_embedding],
                reference_embeddings
            )[0]

            best_similarity = np.max(similarities)

            # ------------------------------------------------
            # PORT OFFICE MATCH
            # ------------------------------------------------

            if best_similarity > SIMILARITY_THRESHOLD:

                # --------------------------------------------
                # DRAW PORT OFFICE BOX
                # --------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

                label = f"PORT OFFICE {best_similarity:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    out.write(frame)

    # --------------------------------------------------------
    # OPTIONAL LIVE VIEW
    # --------------------------------------------------------

    cv2.imshow("Port Office Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

cv2.destroyAllWindows()

print("\n===================================")
print("OUTPUT SAVED:")
print(OUTPUT_PATH)
print("===================================")
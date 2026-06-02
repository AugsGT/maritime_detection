""" 
1. Fix the detection  since floce detects once in 30 frames if by any chance the object isnt detected in the 31st frame then it will have to wait till 60th frame to detect it
2. SAM2 to generate masks for each detected bounding box. not worrking
3. Too much fake detections so set a limit for detection per frame.
4. Include tracking to keep IDs constant throughout the video.
5. Extract the tip of each mask and draw a small circle.  masking not working so obviously tip wont work
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# ------------------------------------------------------------
# Ensure the SAM2 repository can be imported (avoid shadowing)
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "sam2"))

# SAM2 imports – adjust according to installed package structure
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as e:
    raise ImportError(
        "Failed to import SAM2 modules. Ensure the SAM2 repository is installed and its path is added to PYTHONPATH. "
        f"Original error: {e}"
    )

# Configuration
VIDEO_PATH = r"C:\Storage\maritime_final\data\new1.mp4"
OUTPUT_VIDEO = r"C:\\Storage\\maritime_final\\mari\\outputs\\sam2_florence_output8.mp4"

# Florence model and prompt
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
CLASSES = [
    "large ornate building with large arched doorway and red roof",
    "tall white building with balconies and red roof"
]
print(f"Loading Classes{CLASSES}")
#Temporarily hardcoding this need to find alternative later
# Class priority mapping for cross-class NMS. Higher values indicate higher priority.
# Class priority and colors removed – using default visual settings.
DETECTION_INTERVAL = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Global tracking state
next_id = 0
tracks = []  # list of dicts: {"id": int, "tracker": cv2.Tracker, "label": str, "misses": int}

# SAM2 configuration
SAM2_CONFIG = os.path.join(SCRIPT_DIR, "sam2", "config", "sam2.1_hiera_b+.yaml")
SAM2_CHECKPOINT = os.path.join(SCRIPT_DIR, "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")


def load_florence():
    """Load Florence model and processor."""
    print("Loading Florence...")
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID, trust_remote_code=True
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_ID, trust_remote_code=True
    )
    return model, processor

def load_sam2_model():
    """Load SAM2 model and return a predictor instance."""
    print("Loading SAM2 Image Predictor...")
    sam_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam_model)
    return predictor

def detect_florence(frame_bgr, model, processor):

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"

    detections = []

    for cls in CLASSES:

        try:

            inputs = processor(
                text=task_prompt + cls,
                images=image_pil,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=3
                )

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            result = processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=image_pil.size
            )

            result = result.get(task_prompt, {})

            boxes = result.get("bboxes", [])

            for box in boxes:

                detections.append({
                    "box": box,
                    "label": cls
                })

        except Exception as e:

            print(f"Florence failed for {cls}: {e}")

    return detections
def get_mask_for_box(predictor, image_rgb, box):
    """Run SAM2 on a single bounding box and return a binary mask.
    ``box`` is [x1, y1, x2, y2] in pixel coordinates.
    """
    predictor.set_image(image_rgb)
    try:
        out = predictor.predict(box=box, multimask_output=False)
        # SAM2 may return a tuple (masks, scores, ...) or a dict
        if isinstance(out, tuple):
            masks = out[0]
        else:
            masks = out.get("masks", [])
        # masks can be a single mask array or a batch
        if isinstance(masks, np.ndarray):
            mask = masks if masks.ndim == 2 else masks[0]
        else:
            mask = masks[0] if masks else None
    except Exception:
        out = predictor.predict(box=box, multimask_output=True)
        if isinstance(out, tuple):
            masks = out[0]
        else:
            masks = out.get("masks", [])
        mask = masks[0] if masks else None
    if mask is None:
        h, w, _ = image_rgb.shape
        return np.zeros((h, w), dtype=bool)
    return mask.astype(bool)

def create_tracker(frame, box):
    """Create a CSRT tracker compatible with the installed OpenCV version.

    Tries the legacy API first (cv2.legacy.TrackerCSRT_create) and falls back
    to the newer cv2.TrackerCSRT_create if the legacy attribute is unavailable.
    """
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        # Fallback for OpenCV builds where the tracker is exposed at the top level
        tracker = cv2.TrackerCSRT_create()
    x1, y1, x2, y2 = map(int, box)
    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
    return tracker

def draw_mask(frame, mask, color, alpha=0.4):
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[mask] = color
    cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0, frame)

def draw_tip(frame, mask, color=(0, 0, 255)):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return
    tip_x, tip_y = int(xs.min()), int(ys.min())
    cv2.circle(frame, (tip_x, tip_y), 4, color, -1)

# Main processing loop
def main():
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    # Load models
    florence_model, florence_processor = load_florence()
    sam_predictor = load_sam2_model()
    # Persistent tracking structures (global `tracks` and `next_id`).
    # trackers, tracker_colors, tracker_labels are removed.
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Update existing trackers (for continuity when no new detection)

        if frame_idx % DETECTION_INTERVAL == 0:
            # Resize frame for faster SAM2 processing (75% of original size)
            orig_h, orig_w = frame.shape[:2]
            scale = 0.75
            resized_h, resized_w = int(orig_h * scale), int(orig_w * scale)
            resized_frame = cv2.resize(frame, (resized_w, resized_h))
            image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Run Florence detection for all classes
            raw_detections = detect_florence(frame, florence_model, florence_processor)
            # Apply cross‑prompt NMS (no max limit)
            detections = nms_detections(raw_detections, iou_threshold=0.7)
            # Associate detections with existing tracks or create new ones
            match_tracks(detections, tracks, frame)
        else:
            # Draw existing trackers (boxes + label) for continuity
            # Draw existing tracks when no new detection
            for trk in tracks:
                ok, bbox = trk['tracker'].update(frame)
                if ok:
                    x, y, w, h = map(int, bbox)
                    # Update stored box
                    trk['box'] = [x, y, x + w, y + h]
                    # Choose a color based on label hash for consistency
                    color = tuple(int((hash(trk['label']) + i * 50) % 256) for i in (0, 1, 2))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{trk['label']} ID:{trk['id']}", (x, max(30, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    print("\nProcessing completed. Output saved to:", OUTPUT_VIDEO)
def box_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xB <= xA or yB <= yA:
        return 0.0

    inter = (xB - xA) * (yB - yA)

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter)

def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def center_distance(boxA, boxB):
    c1, c2 = box_center(boxA), box_center(boxB)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

# remove_duplicates function removed – NMS now handles suppression
def nms_detections(detections, iou_threshold=0.25):
    """Apply IoU‑based NMS across all class detections.
    All non‑overlapping boxes are kept; no hard limit on count.
    Returns the filtered list of detection dicts."""
    # Sort by box area (largest first) – larger boxes tend to be more stable
    detections = sorted(
        detections,
        key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]),
        reverse=True
    )
    kept = []
    for det in detections:
        suppress = False
        for kept_det in kept:
            iou = box_iou(det["box"], kept_det["box"])
            dist = center_distance(det["box"], kept_det["box"])
            if iou > iou_threshold or dist < 80:
                suppress = True
                break
        if not suppress:
            kept.append(det)
    return kept

def match_tracks(detections, tracks, frame, iou_thresh=0.25, max_misses=10):
    """Match detections to existing tracks or create new tracks.

    Args:
        detections (list): List of detection dicts with 'box' and 'label'.
        tracks (list): Global list of track dicts.
        frame (np.ndarray): Current video frame.
        iou_thresh (float): IoU threshold for matching.
        max_misses (int): Max consecutive misses before removal.
    """
    global next_id
    matched_ids = set()
    for det in detections:
        box = det["box"]
        best_iou = 0
        best_trk = None
        for trk in tracks:
            tbox = trk.get("box")
            if tbox:
                iou = box_iou(box, tbox)
                if iou > best_iou:
                    best_iou = iou
                    best_trk = trk
        if best_iou >= iou_thresh and best_trk and best_trk["id"] not in matched_ids:
            best_trk["tracker"] = create_tracker(frame, box)
            best_trk["box"] = box
            best_trk["label"] = det["label"]
            best_trk["misses"] = 0
            matched_ids.add(best_trk["id"])
        else:
            tracks.append({
                "id": next_id,
                "tracker": create_tracker(frame, box),
                "box": box,
                "label": det["label"],
                "misses": 0,
            })
            next_id += 1
    for trk in tracks:
        if trk["id"] not in matched_ids:
            trk["misses"] += 1
    tracks[:] = [t for t in tracks if t["misses"] <= max_misses]
if __name__ == "__main__":
    main()

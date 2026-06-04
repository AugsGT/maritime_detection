import os
import cv2

VIDEO_PATH = r"C:\Storage\maritime_final\data\fixed1.mp4"

OUTPUT_DIR = r"C:\Storage\maritime_final\output\output_detected11.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    save_path = f"{OUTPUT_DIR}/frame_{frame_id:06d}.jpg"

    cv2.imwrite(save_path, frame)

    print(f"Saved: {save_path}")

    frame_id += 1

cap.release()

print("DONE")
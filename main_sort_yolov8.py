import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

model = YOLO("./weights/best_yolov8.pt")
model.to("cuda")
CONF_THRES = 0.4

tracker = Sort(
    max_age=10,
    min_hits=3,
    iou_threshold=0.3
)

cap = cv2.VideoCapture(
    "D:\\workspace\\Term_7\\Tasarim\\videos\\Visdrone_uav0000009_03358_v.mp4"
)

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRES, device=0, verbose=False)
    dets = []

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        TARGET_CLASS_ID = 1
        for box, score, cls in zip(boxes, scores, classes):
            if int(cls) != TARGET_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box)
            dets.append([x1, y1, x2, y2, score])

    dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))

    tracks = tracker.update(dets)

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("YOLOv8 + SORT", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
total_time = end_time - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0.0

cap.release()
cv2.destroyAllWindows()

print("=" * 40)
print(f"Toplam Frame Sayısı : {frame_count}")
print(f"Toplam Süre (sn)    : {total_time:.2f}")
print(f"Ortalama FPS        : {avg_fps:.2f}")
print("=" * 40)
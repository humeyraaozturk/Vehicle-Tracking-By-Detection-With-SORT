import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from sort.sort import Sort

model = torch.hub.load(
    'yolov5',
    'custom',
    path='/weights/best_yolov5.pt',
    source='local'
)

model.conf = 0.4
model.iou = 0.5

tracker = Sort(
    max_age=10,
    min_hits=3,
    iou_threshold=0.3
)

cap = cv2.VideoCapture("D:\\workspace\\Term_7\\Tasarim\\videos\\Visdrone_uav0000305_00000_v.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    dets = []

    TARGET_CLASS_ID = 1
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, xyxy)
        dets.append([x1, y1, x2, y2, conf])
        if int(cls) != TARGET_CLASS_ID:
            continue

    dets = np.array(dets)

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

    cv2.imshow("YOLOv5 + SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
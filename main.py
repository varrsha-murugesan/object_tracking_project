import cv2
from ultralytics import YOLO
from tracker import CentroidTracker
from stats import StatsCollector
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

tracker = CentroidTracker()
stats = StatsCollector()

video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "output/tracked_output.mp4",
    fourcc,
    20,
    (int(cap.get(3)), int(cap.get(4)))
)

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Detection
    results = model(frame)[0]

    detections = []
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        label = model.names[int(cls)]

        if conf > 0.5:
            detections.append([x1, y1, x2, y2, label])

    # Tracking
    objects = tracker.update(detections)
    stats.update(objects, frame_num)

    # Draw boxes
    for obj_id, (cx, cy, label) in objects.items():
        cv2.putText(frame, f"{label} ID:{obj_id}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    out.write(frame)
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

stats.save_summary()

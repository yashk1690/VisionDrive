import cv2
from ultralytics import YOLO

# -------------------------
# CONFIG
# -------------------------
VIDEO_PATH = "video/eg_1.mp4"
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CONFIDENCE = 0.25
COUNT_LINE_Y = 400  # adjust based on video height

# -------------------------
# LOAD MODEL
# -------------------------
model = YOLO("yolov8s.pt")

# -------------------------
# VIDEO
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

counted_ids = set()
vehicle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # YOLO tracking
    results = model.track(
        frame,
        conf=CONFIDENCE,
        classes=VEHICLE_CLASSES,
        imgsz=960,
        persist=True,
        verbose=False
    )

    # Draw counting line
    cv2.line(frame, (0, COUNT_LINE_Y), (w, COUNT_LINE_Y), (0, 255, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes

        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw center point
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # COUNT LOGIC
            if cy > COUNT_LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

    # Display count
    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Vehicle Counting (YOLO + Tracking)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

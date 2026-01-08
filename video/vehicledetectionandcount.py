import cv2
from ultralytics import YOLO

VIDEO_PATH = "qeg_1.mp4"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.25
COUNT_LINE_RATIO = 0.4

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

print("Video opened:", cap.isOpened())
if not cap.isOpened():
    print("❌ Fix video path first")
    exit()

counted_ids = set()
vehicle_count = 0

# FORCE WINDOW CREATION
cv2.namedWindow("Vehicle Detection + Counting", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    frame = cv2.resize(frame, (640, 360))
    h, w, _ = frame.shape
    COUNT_LINE_Y = int(h * COUNT_LINE_RATIO)

    results = model.track(
        frame,
        conf=CONFIDENCE,
        classes=VEHICLE_CLASSES,
        imgsz=640,
        persist=True,
        verbose=False
    )

    cv2.line(frame, (0, COUNT_LINE_Y), (w, COUNT_LINE_Y), (0, 255, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes

        for box, track_id, conf in zip(
            boxes.xyxy, boxes.id, boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            conf = float(conf)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cy > COUNT_LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id} | {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("Vehicle Detection + Counting", frame)

    # 🔑 THIS MUST RUN EVERY LOOP
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

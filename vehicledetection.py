import cv2
from ultralytics import YOLO

# -------------------------
# CONFIG
# -------------------------
VIDEO_PATH = "video/eg_1.mp4"   # <-- change this


# COCO class IDs
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# -------------------------
# LOAD MODEL
# -------------------------
model = YOLO("yolov8n.pt")  # fast + real-time friendly

# -------------------------
# VIDEO CAPTURE
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# -------------------------
# PROCESS VIDEO
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: resize for FPS boost
    # frame = cv2.resize(frame, (960, 540))

    # YOLO inference
    results = model.track(
        frame,
        conf=0.25,
        classes=VEHICLE_CLASSES,
        persist=True
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Vehicle Detection (Cars | Bikes | Buses | Trucks)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

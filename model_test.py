import time

import cv2
import torch

# Load models
human_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
cylinder_model = torch.hub.load(
    ".", "custom", path=r"C:\Users\srish\yolov5\yolov5\runs\train\clyinders-1dataset3\weights\best.pt", source="local"
)

# Set confidence thresholds
human_confidence_threshold = 0.4
cylinder_confidence_threshold = 0.7

# Cylinder persistence settings
PERSIST_TIME = 2.0  # seconds
previous_cylinders = []

# Load camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    current_time = time.time()

    # Run human detection
    human_results = human_model(frame_resized)
    human_detections = human_results.pandas().xyxy[0]

    # Run cylinder detection
    cylinder_results = cylinder_model(frame_resized)
    cylinder_detections = cylinder_results.pandas().xyxy[0]

    # Draw human boxes
    for _, row in human_detections.iterrows():
        if row["confidence"] > human_confidence_threshold and row["name"] == "person":
            x1, y1, x2, y2 = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, "Human", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Track current high-confidence cylinders
    new_cylinders = []

    for _, row in cylinder_detections.iterrows():
        conf = row["confidence"]
        class_id = int(row["class"])
        label = cylinder_model.names[class_id]

        if label == "gas_cylinder":
            x1, y1, x2, y2 = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            box = (x1, y1, x2, y2)

            if conf >= cylinder_confidence_threshold:
                new_cylinders.append({"box": box, "timestamp": current_time})
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame_resized,
                    f"Gas Cylinder {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

    # Keep old cylinders for persistence
    previous_cylinders = [c for c in previous_cylinders if current_time - c["timestamp"] <= PERSIST_TIME]
    previous_cylinders.extend(new_cylinders)

    # Draw persisted boxes (in purple)
    for c in previous_cylinders:
        x1, y1, x2, y2 = c["box"]
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (200, 50, 255), 2)
        cv2.putText(
            frame_resized, "Gas Cylinder (held)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 255), 2
        )

    # Display the frame
    cv2.imshow("Human + Gas Cylinder Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

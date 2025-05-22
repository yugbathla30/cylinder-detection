import torch
import cv2
import time

# Load YOLOv5n model (nano version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.2  # Confidence threshold

# Open camera
cap = cv2.VideoCapture(0)  # Use 0 for PiCam or USB camera

while True:
    start_time = time.time()  # Reset start_time for each frame

    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing (optional)
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect objects
    results = model(frame_resized)

    # Filter only 'person' and 'truck'
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        if label in ['person']:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Calculate and display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLOv5n Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

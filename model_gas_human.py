import torch
import cv2

# Load models
human_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cylinder_model = torch.hub.load('.', 'custom', path=r'runs\train\clyinders-1dataset4\weights\last.pt', source='local')

# Set confidence threshold
confidence_threshold = 0.4
confidence_threshold_cylinder = 0.5
# Load camera
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640,480))
    
    # Run human detection
    human_results = human_model(frame_resized)
    human_detections = human_results.pandas().xyxy[0]

    # Run cylinder detection
    cylinder_results = cylinder_model(frame_resized)
    cylinder_detections = cylinder_results.pandas().xyxy[0]

    # Draw human boxes (from pretrained model)
    for _, row in human_detections.iterrows():
        if row['confidence'] > confidence_threshold and row['name'] == 'person':
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, 'Human', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw cylinder boxes (from your trained model)
    for _, row in cylinder_detections.iterrows():
        if row['confidence'] > confidence_threshold_cylinder:
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = cylinder_model.names[int(row['class'])]  # 1 → 'gas_cylinder'
            if label == 'gas_cylinder':
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)



    # Display
    cv2.imshow("Human + Gas Cylinder Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


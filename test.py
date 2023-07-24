from ultralytics import YOLO
import cvzone
import cv2
import math

# Running real-time from webcam
cap = cv2.VideoCapture('firevideo.mp4')
model = YOLO('best.pt')

# Reading the classes
classnames = ['fire', 'smoke', 'default']

if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

while True:
    ret, frame = cap.read()  # Read the frame from the video capture
    if not ret:
        print("Error: Could not read the frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)
    cv2.imshow('frame', frame)
    # Getting bbox, confidence, and class names information to work with
    for info in result:
        boxes = info[:, :4]
        scores = info[:, 4]
        classes = info[:, 5].astype(int)

        for box, score, cls in zip(boxes, scores, classes):
            confidence = math.ceil(score * 100)
            if confidence > 50:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[cls]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):  # Press 'q' key to exit the loop and close the window
        break

cap.release()
cv2.destroyAllWindows()

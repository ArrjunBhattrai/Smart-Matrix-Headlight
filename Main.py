import math

from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
model = YOLO("yolov8n.pt")

classNames = model.names
HFoV =60
frame_width = 640
while True:
    success, img = cap.read()
    results = model(img,stream=True)
    for r in results :
        boxes = r.boxes
        for box in boxes :
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2 - x1, y2 - y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            x_center = (x1 + x2) / 2
            X_mid = frame_width / 2
            angle = ((x_center - X_mid) / X_mid) * (HFoV / 2)
            cvzone.putTextRect(img, f'{classNames[cls]} {conf} Angle: {angle:.2f}°',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)


    cv2.imshow("Image", img)
    cv2.waitKey(10)


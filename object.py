import cv2
import numpy as np    

# Load YOLOv4-Tiny Model
net = cv2.dnn.readNet("yolov4-tiny.weights" , "yolov4-tiny.cfg")
# Load COCO class labels
with open("coco.names", "r") as f:classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers().flatten() ]
# Video Source (IP camera)
cap = cv2.VideoCapture('http://192.168.0.104:8080/video')
# Settings
frame_skip = 2
frame_id = 0
colors = np.random.uniform(0 , 255 , size=(len(classes) , 3 ))
# Main Loop
while True:
    ret, frame = cap.read()
    if not ret : break
    start = cv2.getTickCount()
    frame_id += 1 
    if frame_id % frame_skip != 0 : continue
    # Pre-processes frame for YOLO
    small_frame = cv2.resize(frame, (320,320))
    blob = cv2.dnn.blobFromImage(small_frame , 1/255.0 , (416, 416) , swapRB=True)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes, confidences , class_ids = [] , [] , []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id =  np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                x, y, w, h = map(int , detection[:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                boxes.append([x - w // 2, y - h // 2 , w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]: .2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x,y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Calculate FPS
    end = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (end - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Resize Dynamically
    display_frames = cv2.resize(frame, (1200, 800))
    cv2.imshow("YOLOv4 Detection", display_frames)
    if cv2.waitKey(1) == ord('q'): break
cap.release()
cv2.destroyAllWindows()
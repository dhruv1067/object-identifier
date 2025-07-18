import cv2
import numpy as np    
net = cv2.dnn.readNet("yolov4-tiny.weights" , "yolov4-tiny.cfg")
with open("coco.names", "r") as f:classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers().flatten() ]
cap = cv2.VideoCapture('http://192.168.77.207:8080/video')
frame_skip = 5
frame_id = 0
colors = np.random.uniform(0 , 255 , size=(len(classes) , 3 ))
while True:
    ret, frame = cap.read()
    if not ret : break
    frame_id += 1 
    if frame_id % frame_skip != 0 : continue
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
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("YOLOv4 Detection", cv2.resize(frame, (2200, 860)))
    if cv2.waitKey(1) == ord('q'): break
cap.release()
cv2.destroyAllWindows()
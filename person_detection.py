#https://www.youtube.com/watch?v=bUoWTPaKUi4
#Press q to shut down the program
import cv2

#Neural network
net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
#Resize into a small square (320,320) to process a fast analysis
#Scale because the dnn go from 0 to 1 and the pixel value from 0 to 255
model.setInputParams(size=(320,320), scale=1/1)#255)

classes = []
with open("classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        #To get the good shape of inputs
        class_name = class_name.strip()
        classes.append(class_name)

#Camera initialization

cap = cv2.VideoCapture(0)
#Increase camera resolution (Care on rasp)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#HD 1920 x 1080

while True:
    #Receive the frame
    ret, frame = cap.read()
    
    
    #Object detection
    (class_ids, score, bound_boxes) = model.detect(frame)
    
    for class_ids, score, bound_boxes in zip(class_ids, score, bound_boxes):
        x, y, w, h = bound_boxes
        #print(x, y, h, w)
        class_name=classes[int(class_ids)]
        
        if class_name=="person" and score >= 0.8:
        
            cv2.putText(frame, str(class_name), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (200, 0, 50), 3)
    #print("class", class_ids)
    print("scores", score)
    #print("boxes", bound_boxes)
    
    cv2.imshow("Frame", frame)
    
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        cap.release()
        cv2. destroyAllWindows()
        break
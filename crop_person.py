#https://www.youtube.com/watch?v=bUoWTPaKUi4
#Press q to shut down the program
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#Neural network
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
model_crop = cv2.dnn_DetectionModel(net)
#Resize into a small square (320,320) to process a fast analysis
#Scale because the dnn go from 0 to 1 and the pixel value from 0 to 255
model_crop.setInputParams(size=(320,320), scale=1/255)

classes = []
with open("classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        #To get the good shape of inputs
        class_name = class_name.strip()
        classes.append(class_name)


folder_path = "images/"
for image_file in os.listdir(folder_path):
    print(image_file)
    image_path = os.path.join(folder_path, image_file)
    #image = Image.open(image_path)
    image = cv2.imread(image_path)
    # Display the image
    plt.imshow(image)
    plt.title(image_file)
    plt.axis('off')
    plt.show()


    (class_ids, score, bound_boxes) = model_crop.detect(image)
    for class_ids, score, bound_boxes in zip(class_ids, score, bound_boxes):
        x, y, w, h = bound_boxes
        #print(x, y, h, w)
        class_name=classes[int(class_ids)]
        
        if class_name=="person":
        
            #cv2.putText(image, str(class_name)+str(score), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
            #cv2.rectangle(image, (x,y), (x+w,y+h), (200, 0, 50), 3)
            #cv2.imshow("Frame", image)
            #cv2.waitKey(0)
            #print(np.shape(image))
            add = 50
            cropped = np.copy(image[y-add:y+h+add, x-add:x+w+add,:])
            res_cropped = cv2.resize(cropped, (250,250))
            cv2.imshow("NEW", res_cropped)
            cv2.waitKey(0)
            print(np.shape(res_cropped))
            break
    
            cv2.imshow("Frame", image)
            cv2.waitKey(0)

#image = cv2.imread("BIG_DATA/positive/1023.jpg") 
#Object detection
# (class_ids, score, bound_boxes) = model.detect(image)
# for class_ids, score, bound_boxes in zip(class_ids, score, bound_boxes):
#         x, y, w, h = bound_boxes
#         #print(x, y, h, w)
#         class_name=classes[int(class_ids)]
        
#         if class_name=="bottle":
        
#             cv2.putText(image, str(class_name)+str(score), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
#             cv2.rectangle(image, (x,y), (x+w,y+h), (200, 0, 50), 3)
#     #print("class", class_ids)
#             break
    
# cv2.imshow("Frame", image)
# cv2.waitKey(0)
    
    
    
# #new_img=image[y:y+h,x:x+w] 
# #cv2.imwrite(str(idx) + '.png', new_img) 
# """
# while True:
#     #Receive the frame
#     ret, frame = cap.read()
    
    
#     #Object detection
#     (class_ids, score, bound_boxes) = model.detect(frame)
    
#     for class_ids, score, bound_boxes in zip(class_ids, score, bound_boxes):
#         x, y, w, h = bound_boxes
#         #print(x, y, h, w)
#         class_name=classes[int(class_ids)]
        
#         if class_name=="bottle":
        
#             cv2.putText(frame, str(class_name), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
#             cv2.rectangle(frame, (x,y), (x+w,y+h), (200, 0, 50), 3)
#     #print("class", class_ids)
#     #print("scores", score)
#     #print("boxes", bound_boxes)
    
#     cv2.imshow("Frame", frame)

    
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         cap.release()
#         cv2. destroyAllWindows()
#         break
        
# """
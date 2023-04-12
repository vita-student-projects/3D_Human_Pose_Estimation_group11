import h5py
import cv2
import numpy as np

#GET THE IMAGE FROM THE WEBCAM
cap = cv2.VideoCapture(0)
compt = 0
while True:
    compt = compt + 1
    print(compt)
    ret, frame = cap.read()
    cv2.imshow('frame', frame[:,160:, :])
    if cv2.waitKey(1) and 0xFF == ord('q') or compt == 300:
        break
cap.release()
cv2.destroyAllWindows()

cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_shape = np.shape(frame)
data_type = frame.dtype

#SAVE THE IMAGE AS H5
filename = 'test3.h5'
mode = 'w'


with h5py.File(filename, mode) as f:
    dset = f.create_dataset('images', shape = (image_shape), dtype = data_type)
    dset[:] = frame
f.close()

import numpy as np
import cv2
import matplotlib.pyplot as plt
images = np.load("jpg.npy")
csv = np.load("csv.npy")
print(np.shape(images))
print(images[0,:,:,:])
cv2.imshow("images", (images[1,:,:,:]))
cv2.waitKey(0)
# cv2.destroyAllWindows()
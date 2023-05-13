import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def read_csv_file(csv_file):
    data = []
    
    df=np.array(pd.read_csv(csv_file, header=None, index_col=False, dtype=float))
    #print((df))
    return df #data #return_data


def open_img_file(img_file):
    img=cv2.imread(img_file)
    height=256
    width=256
    dim = (height,width)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # resized=np.array(resized,dtype=int)
    return resized


def write_to_CSV(array, file):
    with open(file, 'a', newline='') as file:
        index=array[0]
        pos=array[1]
        img=array[2]
        
        data = {'Column1': index,
        'Column2': pos,
        'Column3': img}
        df = pd.DataFrame(data)
        
        # writer = csv.writer(file)
        # writer.writerow([index,pos]) #,img
        df=pd.read_csv(file, header=None, index_col=False, dtype=float)
        df.to_csv('output.csv', index=False)    


CSV_path = ('DATA_TEST/csv/')
IMG_path = ('DATA_TEST/jpg/')
CSV_write = os.path.join(CSV_path, 'dataset.csv')
open(CSV_write, 'w', newline='')

i=0
size=19312
full_array_csv=np.zeros((size, 17,3))
full_array_jpg=np.zeros((size, 256, 256, 3))

for i in range (0,size): #19312
    CSVfile = os.path.join(CSV_path, '%05d.csv' % (i+1))
    # print(CSVfile)
    data=read_csv_file(CSVfile)
    # print(data)

    IMGfile = os.path.join(IMG_path, '%05d.jpg' % (i+1))
    print(IMGfile)
    img=open_img_file(IMGfile)
    #img = np.transpose(img)

    cv2.imshow("HF",(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows
    full_array_csv[i,:,:] = data
    
    full_array_jpg[i,:,:,:] = (img).copy()/255
    cv2.imshow("images", (full_array_jpg[i,:,:,:]))
    cv2.waitKey(0)

np.save("jpg.npy",full_array_jpg)
np.save("csv.npy",full_array_csv)
cv2.imshow("images", (full_array_jpg[1,:,:,:]))
cv2.waitKey(0)
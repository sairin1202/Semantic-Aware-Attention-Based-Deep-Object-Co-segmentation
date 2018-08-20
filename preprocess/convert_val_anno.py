import numpy as np
import cv2
import glob
import os
from skimage.io import imsave,imread
def pascal_palette():
  palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }

  return palette


im_names = glob.glob(os.path.join('./SegmentationClass', '*.png')) + \
           glob.glob(os.path.join('./SegmentationClass', '*.jpg'))

palette=pascal_palette()

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


count=0
for im_name in im_names:
    image=imread(im_name)
    count+=1
    print("{}/{}".format(count,len(im_names)))
    if len(image.shape)!=2:
        new_image=np.zeros((len(image),len(image[0])))
        for row in range(len(image)):
            for col in range(len(image[0])):
                temp=image[row][col]
                temp=totuple(temp)
                if temp == (0,0,0):
                    new_image[row][col]=0
                    continue
                if temp in palette:
                    new_image[row][col]=palette[temp]
                else:
                    new_image[row][col]=0
        new_image=new_image.astype(np.uint8)
        imsave("label/"+im_name[20:],new_image)
        print(im_name)


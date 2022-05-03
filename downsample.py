import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pyts.image import GramianAngularField

for file in os.scandir("C:/Users/Nirho/Desktop/gasf_800_red/Youtube"):
    #print(file)
    pathnm,exten=os.path.splitext(file)
    #print(pathnm,exten)
    image = cv2.imread(pathnm+exten)
#print("Size of image before pyrDown: ", image.shape)
    image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
#image = cv2.pyrDown(image)
#print("Size of image after pyrDown: ", image.shape)
#cv2.imshow('DownSample', image)
    mypath = os.path.abspath("C:/Users/Nirho/Desktop/gasf_128/Youtube")
    print(pathnm[44:])
    cv2.imwrite((mypath+"/"+pathnm[44:]+".png"), image)

    #resized_image = cv2.resize(image, (100, 50))
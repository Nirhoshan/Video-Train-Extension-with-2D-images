import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
from scipy import stats
import cv2
import os

DATA_DIR = 'generated1'
for file in os.scandir(DATA_DIR):
    #for file in os.scandir(files):
    pathnm, exten = os.path.splitext(file)
    df = pd.read_csv(file,header=None)
    df = df.replace(np.nan, 0)
    data = df.to_numpy(dtype='float')
    pts=[]
    #print(len(data))
    for i in range(len(data)):
          #for j in i:
        pts.append(data[i][i])
    points=[]
    for j in pts:
        points.append((np.sqrt((j+1)/2)))
    #points=pts
    #print(points)
    plt.plot(points)
    mypath = os.path.abspath("C:/Users/Nirho/Desktop/latent/64/Netflix/vid2/features")
    print(pathnm[27:])
    plt.savefig(mypath + "/" + pathnm[27:] + ".png")
    plt.clf()
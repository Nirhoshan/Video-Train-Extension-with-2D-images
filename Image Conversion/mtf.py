
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pyts.image import MarkovTransitionField
import cv2

data_dir = "D:/internship/actual/actual/Youtube"
for file in os.scandir(data_dir):
  pathnm, exten = os.path.splitext(file)
  df = pd.read_csv(file, usecols=[' addr2_bytes'])
  df = df.replace(np.nan, 0)
  data = df.to_numpy(dtype='float')
  pts = []
  for i in data:
    for j in i:
      pts.append(j)
    # print(pts)
  num_of_samples_per_bin = 4
  slice_index = 0
  points = []
  for j in range(int(len(pts) / num_of_samples_per_bin)):
    points.append(np.sum(pts[j * num_of_samples_per_bin:(j + 1) * num_of_samples_per_bin]))

    # print(points)

  X = np.array([points])
    # Compute Gramian angular fields
  mtf = MarkovTransitionField(n_bins=2)
  X_mtf = mtf.fit_transform(X)
    # X_gasf = cv2.resize(X_gasf[0], (64, 64), interpolation=cv2.INTER_AREA)
  mtf_csv = pd.DataFrame(data=X_mtf[0])
  mypath = os.path.abspath("C:/Users/Nirho/Desktop/mtf/dl/Youtube")
  mtf_csv.to_csv(mypath + "/mtf_" + pathnm[36:] + '.csv', header=False, index=False)
  print(pathnm[36:])

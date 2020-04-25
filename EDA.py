import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

img = cv2.imread('score1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sample = gray[:600,:1000]

hist = np.sum(sample, axis=0)

detect = np.zeros_like(hist)
detect[hist < 130000] = 1

plt.plot(range(1000),detect)
plt.show()

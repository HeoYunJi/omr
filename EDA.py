import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

img = cv2.imread('score1.png')

Y, X, C = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scaled= 1 - gray/255

plt.imshow(scaled[100:700,:1000], cmap='gray')
plt.show()

st = scaled[:,int(X*0.1): 2*int(X*0.1)]

for i in range(8):
    st = cv2.bitwise_and(st, scaled[:,(1+i)*int(X*0.1): (2+i)*int(X*0.1)])

plt.imshow(st[100:700], cmap='gray')
plt.show()
    
sample = gray[:600,:1000]

hist = np.sum(sample, axis=0)

detect = np.zeros_like(hist)
detect[hist < 130000] = 1

plt.plot(range(1000),detect)
plt.show()

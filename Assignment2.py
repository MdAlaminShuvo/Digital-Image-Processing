import matplotlib.pyplot as plt
import cv2 
import numpy as np
import math

img = plt.imread('Red.jpg')
grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

row,column = grayscale.shape

s = np.zeros(grayscale.shape, dtype=np.uint8)

T1=50
T2=200

for i in range(row):
    for j in range(column):
        if grayscale[i][j]>=T1 and grayscale[i][j]<=T2:
            s[i][j]=100
        else:
            s[i][j]=10
        
plt.subplot(2,2,1)
plt.title('First Condition')
plt.imshow(s,cmap='gray')
s1 = np.zeros(grayscale.shape, dtype=np.uint8)

for i in range(row):
    for j in range(column):
        if grayscale[i][j]>=T1 and grayscale[i][j]<=T2:
            s1[i][j]=100
        else :
            s1[i][j]=grayscale[i][j]
        


plt.subplot(2,2,2)
plt.title('Second Condition')
plt.imshow(s1,cmap='gray')


s2 = np.zeros(grayscale.shape, dtype=np.uint8)
c = 2
for i in range(row):
    for j in range(column):
        s2[i][j]=c*np.log(1+grayscale[i][j])

plt.subplot(2,2,3)
plt.title('Third Condition')
plt.imshow(s2,cmap='gray')

s3 = np.zeros(grayscale.shape, dtype=np.uint8)
ep = 0.0000001
p = 3

for i in range(row):
    for j in range(column):
        s3[i][j]=c*(grayscale[i][j]+ep)**p

plt.subplot(2,2,4)
plt.imshow(s3,cmap='gray')
plt.show()




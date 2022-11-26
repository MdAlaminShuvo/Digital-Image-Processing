import cv2
import numpy as np
import matplotlib.pyplot as plt


img = plt.imread('Red.jpg')
grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

r, c = grayscale.shape
T1, T2 = 50,120

s = grayscale
for i in range(r):
     for j in range(c):
        if grayscale[i][j] >= T1 and grayscale[i][j] <= T2:
            s[i][j] = 100
        else :
            s[i][j]=10

plt.subplot(2,2,1)
plt.title('First condition Image')
plt.imshow(s, cmap='gray')

s1 = grayscale
for i in range(r):
     for j in range(c):
        if grayscale[i][j] >= T1 and grayscale[i][j] <= T2:
            s1[i][j] = 100
        

plt.subplot(2,2,2)
plt.title('Second condition Image')
plt.imshow(s1, cmap='gray')


c = 2
s2=grayscale
s2 = c *(np.log(1 + grayscale)) 

plt.subplot(2,2,3)
plt.title('Third condition Image')
plt.imshow(s2, cmap='gray')


s3=grayscale
p = 3
elipsion = 1e-6
s3 = c * (grayscale + elipsion) ** p

plt.subplot(2,2,4)
plt.title('Four condition Image')
plt.imshow(s3, cmap='gray')
plt.show()
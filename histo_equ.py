import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


rgb_img = plt.imread('dog.png')



gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

hight, width = gray_img.shape
print(hight,width)


hist = []
# print(hist)

for k in range(256):
    count = 0
    for i in range(hight):
        for j in range(width):
            if gray_img[i][j] == k:
                count = count+1
    hist.append(count)

print(hist) 

pdf = []

for i in range(256):
    pdf.append(hist[i]/(width*hight))

print(pdf)
cdf = []

for i in range(256):
    if i==0:
       cdf.append(pdf[i]) 
    cdf.append((pdf[i]+cdf[i-1]))

# print(cdf)

sk = []

for i in range(256):
    sk.append(cdf[i]*255)


hist_equalize = []

for i in range(256):
    hist_equalize.append(math.ceil(sk[i]))


# print(hist_equalize)

equalize_img = gray

# print(equalize_img)

for i in range(hight):
    for j in range(width):
        equalize_img[i][j] = hist_equalize[gray_img[i][j]]


plt.subplot(1,2,1)
plt.title('old image')
plt.imshow(gray_img, cmap='gray')
plt.subplot(1,2,2)
plt.title('Equalized image')
plt.imshow(equalize_img, cmap='gray')
plt.show()




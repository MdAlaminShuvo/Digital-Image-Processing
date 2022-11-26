import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('./mor.jpg',0)
kernel1 = np.ones((3, 3), dtype=np.uint8)
kernel2 = np.ones((5, 5), dtype=np.uint8)
kernel3 = np.ones((7, 7), dtype=np.uint8) 

th,binary=cv2.threshold(img,127,256,cv2.THRESH_BINARY)
img_erosion1 = cv2.erode(binary, kernel1, iterations=1)
img_dilation1 = cv2.dilate(binary, kernel1, iterations=1)
opening1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)
closing1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1)

img_erosion2 = cv2.erode(binary, kernel2, iterations=1)
img_dilation2 = cv2.dilate(binary, kernel2, iterations=1)
opening2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2)
closing2 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)

img_erosion3 = cv2.erode(binary, kernel3, iterations=1)
img_dilation3 = cv2.dilate(binary, kernel3, iterations=1)
opening3 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3)
closing3 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3)


plt.figure(figsize=(15,15))
plt.subplot(7,2,1)
plt.title("Gray Image")
plt.imshow(binary, cmap='gray')

plt.subplot(7,2,2)
plt.title("Binary Image")
plt.imshow(binary,cmap='gray')

plt.subplot(7,2,3)
plt.title("Erosion1")
plt.imshow(img_erosion1,cmap='gray')

plt.subplot(7,2,4)
plt.title("Erosion2")
plt.imshow(img_erosion2,cmap='gray')

plt.subplot(7,2,5)
plt.title("Erosion3")
plt.imshow(img_erosion3,cmap='gray')

plt.subplot(7,2,6)
plt.title("Dilation1")
plt.imshow(img_dilation1,cmap='gray')

plt.subplot(7,2,7)
plt.title("Dilation2")
plt.imshow(img_dilation2,cmap='gray')

plt.subplot(7,2,8)
plt.title("Dilation3")
plt.imshow(img_dilation3,cmap='gray')

plt.subplot(7,2,9)
plt.title("Opening1")
plt.imshow(opening1,cmap='gray')

plt.subplot(7,2,10)
plt.title("Opening2")
plt.imshow(opening2,cmap='gray')

plt.subplot(7,2,11)
plt.title("Opening3")
plt.imshow(opening3,cmap='gray')

plt.subplot(7,2,12)
plt.title("Closing1")
plt.imshow(closing1,cmap='gray')

plt.subplot(7,2,13)
plt.title("Closing2")
plt.imshow(closing2,cmap='gray')

plt.subplot(7,2,14)
plt.title("Closing3")
plt.imshow(closing3,cmap='gray')

plt.show()
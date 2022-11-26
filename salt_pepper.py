import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


# img_path = 'Red.jpg'
# img1 = plt.imread(img_path)
# img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img = cv2.imread('Red.jpg',
                 cv2.IMREAD_GRAYSCALE)

row , col = img.shape
plt.subplot(3,2,1)
plt.imshow(img,cmap='gray')

number_of_pixel = random.randint(300,10000)

for i in range(number_of_pixel):
    y_coord = random.randint(0,row-1)

    x_coord = random.randint(0,col-1)
    img[y_coord][x_coord] = 255
number_of_pixel = random.randint(300,10000)
for i in range(number_of_pixel):
    y_coord = random.randint(0,row-1)

    x_coord = random.randint(0,col-1)
    img[y_coord][x_coord] = 0
 
plt.subplot(3,2,2)
plt.imshow(img,cmap='gray')

kernel1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
processed_img1 = cv2.filter2D(img, -1, kernel1)

plt.subplot(3,2,3)
plt.imshow(processed_img1,cmap='gray')
plt.show()


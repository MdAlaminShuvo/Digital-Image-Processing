import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


def add_noise(img ):

    
    row , col = img.shape
     
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
       
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
         
        img[y_coord][x_coord] = 255
        
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
       
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
         
        img[y_coord][x_coord] = 0
         
    return img
 
img = cv2.imread('bird.jpg',
                 cv2.IMREAD_GRAYSCALE)
 

gimg = add_noise(img)

plt.subplot(2,2,1)
plt.title('Noisy',fontsize=10)
plt.imshow(gimg,cmap='gray')

kernel1 = np.array([[1/9, 1/9, 1/9],
[1/9 ,1/9, 1/9],
[1/9, 1/9 ,1/9]])
processed_img1 = cv2.filter2D(img, -1, kernel1)

plt.subplot(2,2,2)
plt.title('Average Kernel',fontsize=10)
plt.imshow(processed_img1,cmap='gray')

kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]] , dtype = np.float64)/16
processed_img2 = cv2.filter2D(img, -1, kernel2)

plt.subplot(2,2,3)
plt.title('Gaussian',fontsize=10)
plt.imshow(processed_img2,cmap='gray')

row, col = img.shape

img_new1 = np.zeros([row, col])
 
for i in range(1, row-1):
    for j in range(1, col-1):
        temp = [gimg[i-1, j-1],
               gimg[i-1, j],
               gimg[i-1, j + 1],
               gimg[i, j-1],
               gimg[i, j],
               gimg[i, j + 1],
               gimg[i + 1, j-1],
               gimg[i + 1, j],
               gimg[i + 1, j + 1]]
         
        temp = sorted(temp)
        img_new1[i, j]= temp[4]
 
img_new1 = img_new1.astype(np.uint8)



plt.subplot(2,2,4)
plt.title('Median',fontsize=10)
plt.imshow(img_new1,cmap='gray')

plt.show()



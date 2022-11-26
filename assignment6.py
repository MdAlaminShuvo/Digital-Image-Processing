import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random

def main():
    rgb=plt.imread('Red.jpg')
    plt.figure(figsize=(50,100))
    grayscale=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    plt.subplot(3,2,1)
    plt.title("Grayscale Image",fontsize=10)
    plt.imshow(grayscale,cmap='gray')
    
    w,h=grayscale.shape
    
    #Filtered Image(averaging)
    average_kernel = np.array([[1/9, 1/9, 1/9],
                        [1/9 ,1/9, 1/9],
                        [1/9, 1/9 ,1/9]])
    average_img = cv2.filter2D(grayscale, -1,average_kernel)

    plt.subplot(3,2,2)
    plt.title("Filtered Image(averaging)",fontsize=10)
    plt.imshow(average_img,cmap='gray')
    
    #Nosiy Image
    nosiy_img=grayscale
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
       
        y_coord=random.randint(0, w - 1)
        x_coord=random.randint(0, h - 1)
        nosiy_img[y_coord][x_coord] = 255
        
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
       
        y_coord=random.randint(0, w - 1)
        x_coord=random.randint(0, h - 1)
        nosiy_img[y_coord][x_coord] = 0
    
    
    plt.subplot(3,2,3)
    plt.title("Nosiy Image(salt & pepper noise)",fontsize=10)
    plt.imshow(nosiy_img,cmap='gray')
    
    
    #Nosiy Image(average filtering)
    average_kernel = np.array([[1/9, 1/9, 1/9],
                        [1/9 ,1/9, 1/9],
                        [1/9, 1/9 ,1/9]])
    nosiy_average_img = cv2.filter2D(nosiy_img, -1,average_kernel)

    plt.subplot(3,2,4)
    plt.title("Filtered Nosiy Image(averaging)",fontsize=10)
    plt.imshow(nosiy_average_img,cmap='gray')
    
    #Gaussian Filter
    gaussian_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]],dtype =np.float64)/16
    nosiy_gaussian_img= cv2.filter2D(nosiy_img, -1,gaussian_kernel)
    
    plt.subplot(3,2,5)
    plt.title('Filtered Nosiy Image(gaussian kernel)',fontsize=10)
    plt.imshow(nosiy_gaussian_img,cmap='gray')
    
    #Median Filter 
    median_img = cv2.medianBlur(nosiy_img,3)

    plt.subplot(3,2,6)
    plt.title('Median Filter',fontsize=10)
    plt.imshow(median_img,cmap='gray')
    
    plt.show()
    
if __name__=='__main__':
    main()
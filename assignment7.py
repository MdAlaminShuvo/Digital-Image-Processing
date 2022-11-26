import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    img_path='Red.jpg'
    img=plt.imread(img_path)
    
    

    
    grayscale=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    w,h =grayscale.shape
    plt.subplot(4,2,1)
    plt.title("Grayscale")
    plt.imshow(grayscale,cmap='gray')
    
    plt.subplot(4,2,2)
    plt.title("grayscale Histrogram")
    plt.hist(grayscale.ravel(),255,[0,255])
    
    left=grayscale
    for i in range(0,w):
        for j in range(0,h):
            left[i][j] = grayscale[i][j] - 50
            if left[i][j]<0:
                left[i][j]=0
    plt.subplot(4,2,3)
    plt.title("Left")
    plt.imshow(left,cmap='gray')
    
    plt.subplot(4,2,4)
    plt.title("Left Histrogram")
    plt.hist(left.ravel(),255,[0,255])
    
    right=grayscale
    for i in range(0,w):
        for j in range(0,h):
            right[i][j] = grayscale[i][j] + 80
            if right[i][j]>255:
                right[i][j]=255
    plt.subplot(4,2,5)
    plt.title("Right")
    plt.imshow(right,cmap='gray')
    
    plt.subplot(4,2,6)
    plt.title("Right Histrogram")
    plt.hist(right.ravel(),255,[0,255])
    
    middle=grayscale
    for i in range(0,w):
        for j in range(0,h):
            middle[i][j] =(grayscale[i][j] + 127)/2
            if middle[i][j]>255:
                middle[i][j]=175
            if middle[i][j]<100:
                middle[i][j]=100
    plt.subplot(4,2,7)
    plt.title("Specific_Range")
    plt.imshow(middle,cmap='gray')
    
    plt.subplot(4,2,8)
    plt.title("Specific_Range Histrogram")
    plt.hist(middle.ravel(),255,[0,255])
    
    
    plt.show()
    
if __name__=='__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math


def main():
    
    x, y = 3, 2
    pos, channel = 1, 0
    
    img_path= 'Red.jpg'
    rgb=plt.imread(img_path)
    print(rgb)
    print(rgb.shape)
    grayscale = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

    plt.figure(figsize=(50,100))
    plt.subplot(x,y,pos)
    plt.title('Grayscale')
    plt.imshow(grayscale,cmap='gray')

    
    
    '''Question 1-: Comparison between built in and Custom histogram'''
    # find histogram by own process
    img = np.array(grayscale)
    img = img.reshape(-1);
    yaxis = np.zeros((256,),dtype=int)
    for i in range(0,len(img)):
        yaxis[img[i]] += 1
    
    #print(y)
    xaxis = np.array([i for i in range(0,256)])
    #print(x.shape,y.shape)
    plt.subplot(x,y,pos+2)
    plt.title('Customize Histogram')
    plt.plot(x,y)
    plt.bar(xaxis,yaxis,width=1.001)
    
    #built in histogram
    plt.subplot(x,y,pos+3)
    plt.title('Built In Histogram')
    plt.hist(img.ravel(),256,[0,256]);
    
    
    
    '''Question2-: Comparison between built in kernel function and custom kernel'''
    kernel = np.array([
        [3,0,-3],
        [10,0,-10],
        [3,0,-3]
    ])
    print("Values of kernal={}".format(kernel))
    
    grayscale = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    
    kernel_output = cv.filter2D(grayscale,-1,kernel)
    plt.subplot(x,y,pos+5)
    plt.title('Built In Convolution Kernel/Filter')
    plt.imshow(kernel_output,cmap='gray')
    
   
    img=grayscale
    w, h = img.shape
    new_img = np.zeros(shape=(w+2,h+2))
    w, h = new_img.shape
    new_img[1:w-1,1:h-1] = img
    new_img.astype(int)
    
    img=new_img
    _, k = kernel.shape
    w, h = img.shape
    new_w, new_h = w-k+1, h-k+1
    conv_img = np.zeros(shape=(new_w,new_h))
    for i in range(new_w):
        for j in range(new_h):
            mat = img[i:i+k,j:j+k]
            conv_img[i,j] = np.sum(np.multiply(kernel,mat))
            if conv_img[i,j]<0:
                conv_img[i,j]=0
            elif(conv_img[i,j]>255):
                conv_img[i,j]=255
    
    my_kernel_output = conv_img
    plt.subplot(x,y,pos+4)
    plt.title('Customize Convolution Kernel/Filter')
    plt.imshow(my_kernel_output,cmap='gray')
    
    
    plt.show()
    
if __name__ == '__main__':
    main()

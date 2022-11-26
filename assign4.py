import matplotlib.pyplot as plt
import cv2
import numpy as np


def main():
    img_path = 'Red.jpg'
    rgb = plt.imread(img_path)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    plt.subplot(3,2,1)
    plt.title('RGB')
    plt.imshow(rgb)

    plt.subplot(3,2,2)
    plt.title('Grayscale')
    plt.imshow(grayscale,cmap='gray')

    '''Implement histogram and compare the result with built-in histogram function.'''
    img = np.array(grayscale)
    img = img.reshape(-1)
    yAxis = np.zeros((256,),dtype=int)
    for i in range(len(img)):
        yAxis[img[i]] += 1
    

    xAxis = np.array([i for i in range(0,256)])
    #print(xAxis.shape,yAxis.shape)
    plt.subplot(3,2,3)
    plt.title('Implement Histogram function')
    plt.bar(xAxis,yAxis,width=1)

    plt.subplot(3,2,4)
    plt.title('Built in histogram function')
    plt.hist(grayscale.ravel(),256,[0,256])


    '''Implement neighborhood processing and compare the result with built-in cv2.filter2D.'''
    kernel = np.array([[0, -1, 0], [-1, 4, 1], [0, -1, 0]])
    grayscale2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel_output = cv2.filter2D(grayscale2,-1,kernel)
    plt.subplot(3,2,5)
    plt.title('Built in cv2.filter2D')
    plt.imshow(kernel_output, cmap='gray')

    img2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    w, h = img2.shape
    new_img = np.zeros(shape=(w+2,h+2))
    w, h = new_img.shape
    new_img[1:w-1,1:h-1] = img2
    new_img.astype(int)

    img2 = new_img
    w, h = img2.shape
    _, k = kernel.shape
    new_w, new_h = w-k+1, h-k+1
    convoluted_img = np.zeros(shape=(new_w,new_h))
    for i in range(new_w):
        for j in range(new_h):
            val = img2[i:i+k,j:j+k]
            convoluted_img[i,j] = np.sum(np.multiply(kernel,val))
            if convoluted_img[i,j]<0:
                convoluted_img[i,j]=0
            elif convoluted_img[i,j]>255:
                convoluted_img[i,j]=255

    plt.subplot(3,2,6)
    plt.title('Implement neighborhood processing')
    plt.imshow(convoluted_img, cmap='gray')



    plt.show()


if __name__ == '__main__':
    main()
from matplotlib import pyplot as plt
import cv2
import numpy as np


def main():

    img_path = 'Red.jpg'
    rgb = plt.imread(img_path)
    print(rgb.shape)


    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(grayscale.shape, grayscale.max(), grayscale.min())

    plt.subplot(6, 3, 1)
    plt.title('Grayscale')
    plt.imshow(grayscale, cmap = 'gray')


    kernel1 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)
    #print(kernel2, kernel2.dtype, kernel2.size)
    processed_img = cv2.filter2D(grayscale, -1, kernel1)

    plt.subplot(6, 3, 2)
    plt.title('Kernel')
    plt.imshow(processed_img, cmap = 'gray')

    kernel2 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)
    #print(kernel2, kernel2.dtype, kernel2.size)
    processed_img2 = cv2.filter2D(grayscale, -1, kernel2)

    plt.subplot(6, 3, 3)
    plt.title('Kernel2')
    plt.imshow(processed_img2, cmap = 'gray')

    kernel3 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)
    #print(kernel2, kernel2.dtype, kernel2.size)
    processed_img3 = cv2.filter2D(grayscale, -1, kernel3)

    plt.subplot(6, 3, 4)
    plt.title('Kernel3')
    plt.imshow(processed_img3, cmap = 'gray')

    kernel4 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)
    #print(kernel2, kernel2.dtype, kernel2.size)
    processed_img4 = cv2.filter2D(grayscale, -1, kernel4)

    plt.subplot(6, 3, 5)
    plt.title('Kernel4')
    plt.imshow(processed_img4, cmap = 'gray')

    kernel5 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)
    #print(kernel2, kernel2.dtype, kernel2.size)
    processed_img5 = cv2.filter2D(grayscale, -1, kernel5)

    plt.subplot(6, 3, 6)
    plt.title('Kernel5')
    plt.imshow(processed_img5, cmap = 'gray')

    kernel6 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)
    #print(kernel2, kernel2.dtype, kernel2.size)
    processed_img6 = cv2.filter2D(grayscale, -1, kernel6)

    plt.subplot(6, 3, 7)
    plt.title('Kernel6')
    plt.imshow(processed_img6, cmap = 'gray')
    plt.show()
    plt.figure(figsize=(40, 40))
    plt.subplots_adjust(
                    left=0.1,
                    bottom=0.9,
                    right=0.5, 
                    top=1.9, 
                    wspace=0.9, 
                    hspace=1.3)
    


if __name__ == '__main__':
	main()
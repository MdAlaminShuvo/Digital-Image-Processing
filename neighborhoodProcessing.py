from matplotlib import pyplot as plt
import cv2
import numpy as np


def main():

    img_path = 'image/1.jpg'
    rgb = plt.imread(img_path)
    print(rgb.shape)


    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(grayscale.shape, grayscale.max(), grayscale.min())

    kernel1 = np.ones((3, 3), dtype=np.int8)
    processed_img = cv2.filter2D(grayscale, -1, kernel1)

    kernel2 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)
    #print(kernel2, kernel2.dtype, kernel2.size)
    processed_img2 = cv2.filter2D(grayscale, -1, kernel2)

    img_set = [grayscale, processed_img, processed_img2 ]
    title_set = ['Grayscale', 'kernel1', 'kernel2']
    plot_img(img_set, title_set)



def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize=(20, 20))
    for i in range(n):
            plt.subplot(2, 3, i+1)
            plt.title(title_set[i])
            plt.imshow(img_set[i], cmap = 'gray')
    plt.show()


if __name__ == '__main__':
    main()
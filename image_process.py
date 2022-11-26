import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('Red.jpg', 0)


lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
         lst.append(np.binary_repr(img[i][j] ,width=8)) 


eight_bit_img = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(img.shape[0],img.shape[1])
seven_bit_img = (np.array([int(i[1]) for i in lst],dtype = np.uint8) * 64).reshape(img.shape[0],img.shape[1])
six_bit_img = (np.array([int(i[2]) for i in lst],dtype = np.uint8) * 32).reshape(img.shape[0],img.shape[1])
five_bit_img = (np.array([int(i[3]) for i in lst],dtype = np.uint8) * 16).reshape(img.shape[0],img.shape[1])
four_bit_img = (np.array([int(i[4]) for i in lst],dtype = np.uint8) * 8).reshape(img.shape[0],img.shape[1])
three_bit_img = (np.array([int(i[5]) for i in lst],dtype = np.uint8) * 4).reshape(img.shape[0],img.shape[1])
two_bit_img = (np.array([int(i[6]) for i in lst],dtype = np.uint8) * 2).reshape(img.shape[0],img.shape[1])
one_bit_img = (np.array([int(i[7]) for i in lst],dtype = np.uint8) * 1).reshape(img.shape[0],img.shape[1])


plt.subplot(4, 2, 1)
plt.title('Eight Bit')
plt.imshow(eight_bit_img, cmap = 'gray')

plt.subplot(4, 2, 2)
plt.title('Seven Bit')
plt.imshow(seven_bit_img, cmap = 'gray')

plt.subplot(4, 2, 3)
plt.title('Six Bit')
plt.imshow(six_bit_img, cmap = 'gray')

plt.subplot(4, 2, 4)
plt.title('Fifth Bit')
plt.imshow(five_bit_img, cmap = 'gray')

plt.subplot(4, 2, 5)
plt.title('Four Bit')
plt.imshow(four_bit_img, cmap = 'gray')

plt.subplot(4, 2, 6)
plt.title('Three Bit')
plt.imshow(three_bit_img, cmap = 'gray')


plt.subplot(4, 2, 7)
plt.title('Two Bit')
plt.imshow(two_bit_img, cmap = 'gray')

plt.subplot(4, 2, 8)
plt.title('One Bit')
plt.imshow(one_bit_img, cmap = 'gray')

plt.show()
cv2.waitKey(0) 
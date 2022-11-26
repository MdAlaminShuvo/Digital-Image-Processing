import matplotlib.pyplot as plt
import cv2
import mam

# mam.main()
def main():
    img_path = "C:/Users/Dell/Desktop/DIP/Red.jpg"
    
    img = cv2.imread(img_path)
    print(img.shape)

    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)

    plt.imshow(img)
    plt.subplot(2,2,1)
    
    plt.imshow(img[:,:,0], cmap='gray')
    plt.subplot(2,2,2)
    
    plt.imshow(img[:,:,1], cmap='gray')
    plt.subplot(2,2,3)
    
    plt.imshow(img[:,:,2], cmap='gray')
    plt.subplot(2,2,4)
    
    plt.imshow(img)


    plt.show()

if __name__ == '__main__':
    main()
    # print('ok')
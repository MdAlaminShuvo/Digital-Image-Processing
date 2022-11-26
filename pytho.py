
import matplotlib.pyplot as plt
import cv2



def main():
    '''	Load an RGB image.	'''
    img_path = "C:\\Users\\Dell\\Desktop\\DIP\\Red.jpg"
    print(img_path)
    rgb = plt.imread(img_path) 
    print(rgb.shape, rgb.max(), rgb.min())
    print(rgb)
    '''	Split loaded RGB image into red, green and blue channels. '''

    plt.subplot(1,1,1)
    plt.title('Red channel')
    red = rgb[:, :, 0]
    plt.imshow(red, cmap='Reds')
    
    plt.show()

if __name__ == '__main__':
	main()
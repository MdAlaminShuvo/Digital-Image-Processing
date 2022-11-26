
import matplotlib.pyplot as plt
import cv2



def main():
    
    img_path = "C:\\Users\\Dell\\Desktop\\DIP\\Red.jpg"
    print(img_path)
    rgb = plt.imread(img_path)
    
    plt.subplot(6,2,1)
    plt.title('RGB')
    plt.imshow(rgb)

    plt.subplot(6,2,2)
    plt.title('RGB Histogram')
    plt.hist(rgb.ravel(),256,[0,256])
    
    plt.subplot(6,2,3)
    plt.title('Red channel')
    red = rgb[:, :, 0]
    plt.imshow(red, cmap='Reds')
    
    plt.subplot(6,2,4)
    plt.title('Red Histogram')
    plt.hist(red.ravel(),256,[0,256])
    
    plt.subplot(6,2,5)
    plt.title('Green channel')
    green = rgb[:, :, 1]
    plt.imshow(green,cmap='Greens')
    
    plt.subplot(6,2,6)
    plt.title('Green Histogram')
    plt.hist(green.ravel(),256,[0,256]);
    
    plt.subplot(6,2,7)
    plt.title('Blue channel')
    blue = rgb[:, :, 2]
    plt.imshow(blue,cmap='Blues')
    
    plt.subplot(6,2,8)
    plt.title('Blue Histogram')
    plt.hist(blue.ravel(),256,[0,256])
    
    
    plt.subplot(6,2,9)
    plt.title('Grayscale channel')
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    plt.imshow(grayscale,cmap='gray')
     
    plt.subplot(6,2,10)
    plt.title('Grascale Histogram')
    plt.hist(grayscale.ravel(),256,[0,256])
    
    plt.subplot(6,2,11)
    plt.title('Binary channel')
    th, binary_img = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    plt.imshow(binary_img,cmap='binary')
    
    plt.subplot(6,2,12)
    plt.title('Binary Histogram')
    plt.hist(binary_img.ravel(),256,[0,256]);
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.8, 
                    top=0.9, 
                    wspace=0.9, 
                    hspace=1.1)
    
    plt.show()
    #plt.savefig('Dell/Figure_1')
 
    

if __name__ == '__main__':
	main()
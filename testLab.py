from concurrent.futures import process
import matplotlib.pyplot as plt
import numpy as np

def main():
    img_path = 'Red.jpg'
    rgb = plt.imread(img_path)
    red = rgb[:,:,0]
    green = rgb[:,:,1]
    blue = rgb[:,:,2]
    
    r,c = red.shape
    gray = np.zeros(red.shape, dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            gray[i][j] = int(0.299*red[i][j]+0.587*green[i][j]+0.114*blue[i][j])


    process_img = [rgb,gray,red,green,blue]
    img_title = ['RGB','GRAY','RED','GREEN','BLUE']
    n=len(process_img)
    for i in range(n):
        img = process_img[i]
        ch = len(img.shape)
        plt.subplot(3,2,i+1)
        if (ch==3):
            plt.imshow(process_img[i])
        else:
            plt.imshow(process_img[i],cmap='gray')
        plt.title(img_title[i],fontsize=10)

    
    
    plt.show()

main()
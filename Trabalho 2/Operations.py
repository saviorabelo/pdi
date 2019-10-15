import cv2
import numpy as np
import matplotlib.pyplot as plt

class Operations:

    def __init__(self, image, name_file):
        # Check if the image has only two channels
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            self.image = image
        self.name_file = name_file[:-4]

    def thresholdMean(self):
        media = np.mean(self.image)
        img_bin = ((self.image >= media) * 255).astype('uint8')
        
        self.plotResult(img_bin, 'Binary')
        return img_bin, media
    
    def iterativeMean(self):
        img = self.image
        mean_initial = np.mean(img)
        while 1:
            mean_1 = np.mean(img[img < mean_initial])        
            mean_2 = np.mean(img[img >= mean_initial])
            
            media_new = (mean_1 + mean_2) / 2
            if media_new.astype("uint8") == mean_initial.astype("uint8"):
                break
            else:
                mean_initial = media_new

        img_bin = ((img > media_new) * 255).astype("uint8")
        self.plotResult(img_bin, 'Iterative Mean: {:.1f}'.format(media_new))
        #return img_bin, media_new
    
    def plotResult(self, result, title):
        fig = plt.figure(figsize=(9,3), dpi=80)
        a = fig.add_subplot(1,2,1)
        a.axis('off')
        plt.imshow(self.image, cmap=plt.get_cmap('gray'))
        a.set_title('Original')

        a = fig.add_subplot(1,2,2)
        a.axis('off')
        plt.imshow(result, cmap=plt.get_cmap('gray'))
        a.set_title(title)

        plt.show()




    

import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.morphology import disk

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
        self.plotResult(img_bin, 'Iterative Mean: {:.2f}'.format(media_new))
        #return img_bin, media_new
    
    def watershed(self):
        image = self.image
        image_ext = morphology.dilation(image, disk(5)) - image
        
        m,n = image.shape
        markers = np.zeros([m,n])
        # Center
        m = int(m/2)
        n = int(n/2)
        markers[20:40,20:40] = 200
        markers[m:m+20,n:n+20] = 100
        
        ws = morphology.watershed(image_ext, markers)
        self.plotWatershed(image, 255-image_ext, markers, ws)

    def plotWatershed(self, image, dilation, markers, watershed):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(8, 2.5), sharex=True, sharey=True)

        ax0.imshow(image, cmap=plt.get_cmap('gray'))
        ax0.set_title('Original')
        ax0.axis('off')

        ax1.imshow(dilation, cmap=plt.get_cmap('gray'))
        ax1.set_title('External Dilation')
        ax1.axis('off')

        ax2.imshow(markers, cmap=plt.get_cmap('gray'))
        ax2.set_title('Markers')
        ax2.axis('off')
        
        ax3.imshow(watershed, cmap=plt.get_cmap('nipy_spectral'), interpolation='nearest')
        ax3.set_title('Watershed')
        ax3.axis('off')

        fig.tight_layout()
        plt.show()

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

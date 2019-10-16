import cv2
import numpy as np
from collections import deque
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
    
    def plotResult(self, result, title):
        fig = plt.figure(figsize=(8, 2.5), dpi=80)
        a = fig.add_subplot(1,2,1)
        a.axis('off')
        plt.imshow(self.image, cmap=plt.get_cmap('gray'))
        a.set_title('Original')

        a = fig.add_subplot(1,2,2)
        a.axis('off')
        plt.imshow(result, cmap=plt.get_cmap('gray'))
        a.set_title(title)

        plt.show() 
    
    def getSeed(self, w, h):
        p = 10.
        
        pos_ini_x_mrk = int(w/2 - p*w/100.)
        pos_ini_y_mrk = int(h/2 - p*h/100.)
        pos_fim_x_mrk = int(w/2 + p*w/100.)
        pos_fim_y_mrk = int(h/2 + p*h/100.)

        seed = np.zeros(shape=(w,h), dtype=np.uint8)
        seed[pos_ini_x_mrk:pos_fim_x_mrk, pos_ini_y_mrk:pos_fim_y_mrk] = 255
        
        return seed

    def neighbors(self, x, y, w, h):
        points = [(x-1,y), (x+1, y), (x,y-1), (x,y+1),
                (x-1,y+1), (x+1, y+1), (x-1,y-1), (x+1,y-1)]
        
        list_ = deque()
        for p in points:
            if (p[0]>=0 and p[1]>=0 and p[0]<w and p[1]<h):
                list_.append(p)
                
        return list_
        
    def regionGrowth(self, epsilon=10):
        image = cv2.blur(self.image, (5,5))
        w, h = image.shape
        
        reg = self.getSeed(w, h)
        queue = deque()
        for x in range(w):
            for y in range(h):
                if reg[x, y] == 255:
                    queue.append((x, y))
        
        while queue:
            point = queue.popleft()
            x = point[0]
            y = point[1]

            v_list = self.neighbors(x, y, w, h)
            for v in v_list:
                v_x = v[0]
                v_y = v[1]
                if((reg[v_x][v_y] != 255) and (abs(int(image[x][y])-int(image[v_x][v_y])) < epsilon)):
                    reg[v_x][v_y] = 255
                    queue.append(v)
            
        self.plotResult(reg, 'Region Growth')
        #return reg

    def getMarkers(self, m, n):
        markers = np.zeros([m,n])
        # Center
        m = int(m/2)
        n = int(n/2)
        markers[20:40,20:40] = 200
        markers[m:m+20,n:n+20] = 100

        return markers

    def watershed(self):
        image = self.image
        image_ext = morphology.dilation(image, disk(5)) - image
        
        m, n = image.shape
        markers = self.getMarkers(m, n)
        
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

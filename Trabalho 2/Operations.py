import cv2
import pywt
import numpy as np
from collections import deque
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.morphology import disk, square, cube, diamond, rectangle, ball, star

class Operations:

    def __init__(self, image, name_file):
        # Check if the image has only two channels
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            self.image = image
        self.name_file = name_file[:-4]

    def translation(self):
        rows, cols = self.image.shape

        M = np.float32([[1,0,100],[0,1,50]])
        trans = cv2.warpAffine(self.image, M, (cols, rows))

        self.plotResult(trans, 'Translation')
    
    def scaling(self):
        rows, cols = self.image.shape
        sca = cv2.resize(self.image, (2*cols, 2*rows), interpolation = cv2.INTER_CUBIC)

        self.plotResult(sca, 'Scaling')
        cv2.imwrite('./Results/{}-scaling.png'.format(self.name_file), sca)
    
    def rotation(self):
        rows, cols = self.image.shape

        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        rot = cv2.warpAffine(self.image, M, (cols, rows))

        self.plotResult(rot, 'Rotation')
        #cv2.imwrite('./Results/{}-rotation.png'.format(self.name_file), rot)
    
    def dilation(self):
        image = self.image
        se = ball(5)
        dil = morphology.dilation(image, se) - image

        self.plotResult(dil, 'Dilation')
        #cv2.imwrite('./Results/{}-dilation.png'.format(self.name_file), dil)
    
    def erosion(self):
        image = self.image
        se = disk(5)
        ero = morphology.erosion(image, se) - image

        self.plotResult(ero, 'Erosion')
        #cv2.imwrite('./Results/{}-erosion.png'.format(self.name_file), ero)
    
    def kmeans(self):
        img = self.image
        Z = img.reshape((-1,2))
        Z = np.float32(Z)

        K = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        aux = center[label.flatten()]
        result = aux.reshape((img.shape))

        self.plotResult(result, 'Kmeans (K={})'.format(K))
        #cv2.imwrite('./Results/{}-kmeans.png'.format(self.name_file), result)
    
    def wavelet(self):
        img_haar = pywt.dwt2(self.image, 'haar')
        cA, (cH, cV, cD) = img_haar

        plt.figure(figsize=(9,9))

        plt.subplot(221)
        plt.imshow(cA, 'gray')
        plt.title('Original')
        plt.subplot(222)
        plt.imshow(cH, 'gray')
        plt.title('Horizontais')
        plt.subplot(223)
        plt.imshow(cV, 'gray')
        plt.title('Verticais')
        plt.subplot(224)
        plt.imshow(cD, 'gray')
        plt.title('Diagonais')
        plt.show()
    
    def fourier(self):
        img = self.image
        rows, cols = img.shape
        img_dft = np.fft.fft2(img)
        img_dft_shift = np.fft.fftshift(img_dft)
        img_dft_mag = np.abs(img_dft_shift)

        self.plotResult(np.log(img_dft_mag), 'Espectro em frequência')

        img_idft = np.fft.ifft2(img_dft)
        img_inversa = np.abs(img_idft)

        self.plotResult(img_inversa, 'Imagem após IDFT')

        #cv2.imwrite('./Results/{}-kmeans.png'.format(self.name_file), result)

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

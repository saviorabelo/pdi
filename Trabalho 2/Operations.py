import cv2
import pywt
import numpy as np
from collections import deque
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.morphology import disk, square, star

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
        cv2.imwrite('./Results/{}-translation.png'.format(self.name_file), trans)

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
        cv2.imwrite('./Results/{}-rotation.png'.format(self.name_file), rot)
    
    # plot structuring element
    def se(self):

        fig = plt.figure(figsize=(10, 5))
        a = fig.add_subplot(1, 3, 1)
        plt.imshow(disk(5), cmap=plt.cm.gray)
        a.set_title('Disk 5px ')
        plt.axis('off')

        a = fig.add_subplot(1, 3, 2)
        plt.imshow(square(5), cmap=plt.cm.gray)
        a.set_title('Square 5px')
        plt.axis('off')

        a = fig.add_subplot(1, 3, 3)
        plt.imshow(star(5), cmap=plt.cm.gray)
        a.set_title('Star 5px')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('./Results/se.png')
        plt.show()
        
    
    def dilation(self):
        image = self.image
        se = star(5)
        dil = morphology.dilation(image, se)

        self.plotResult(dil, 'Dilation')
        #cv2.imwrite('./Results/{}-dilation-star5.png'.format(self.name_file), dil)
    
    def erosion(self):
        image = self.image
        se = star(5)
        ero = morphology.erosion(image, se)

        self.plotResult(ero, 'Erosion')
        cv2.imwrite('./Results/{}-erosion-star5.png'.format(self.name_file), ero)

    def morphologicalGradient(self):
        image = self.image

        se = disk(5)
        dil = morphology.dilation(image, se)
        ero = morphology.erosion(image, se)
        mg = dil - ero
        self.plotResult(mg, 'Morphological Gradient')
        #cv2.imwrite('./Results/{}-gradient.png'.format(self.name_file), mg)

        internal = image - ero
        self.plotResult(internal, 'Internal Gradient')
        #cv2.imwrite('./Results/{}-gradient-internal.png'.format(self.name_file), internal)

        external = dil - image
        self.plotResult(external, 'External Gradient')
        #cv2.imwrite('./Results/{}-gradient-external.png'.format(self.name_file), external)
    
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
        cv2.imwrite('./Results/{}-region-growth.png'.format(self.name_file), reg)
    
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

        self.plotResult(result, 'K-means (K={})'.format(K))
        cv2.imwrite('./Results/{}-k-means-{}.png'.format(self.name_file, K), result)

    def houghCircles(self):
        img = self.image
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=50, param2=30, minRadius=10, maxRadius=40)

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow('Detected Circles',cimg)
        cv2.imwrite('./Results/{}-houghCircles.png'.format(self.name_file), cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def houghLines(self):
        dst = cv2.Canny(self.image, 50, 200, None, 3)
    
        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv2.imwrite('./Results/{}-houghLines.png'.format(self.name_file), cdst)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

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

        cv2.imwrite('./Results/{}-ws-external-dilation.png'.format(self.name_file), 255-image_ext)
        cv2.imwrite('./Results/{}-ws-markers.png'.format(self.name_file), markers)
        cv2.imwrite('./Results/{}-ws.png'.format(self.name_file), ws)
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

        cv2.imwrite('./Results/{}-wavelet-cA.png'.format(self.name_file), cA)
        cv2.imwrite('./Results/{}-wavelet-cH.png'.format(self.name_file), cH)
        cv2.imwrite('./Results/{}-wavelet-cV.png'.format(self.name_file), cV)
        cv2.imwrite('./Results/{}-wavelet-cD.png'.format(self.name_file), cD)
    
    def fourier(self):
        img = self.image
        rows, cols = img.shape
        img_dft = np.fft.fft2(img)
        img_dft_shift = np.fft.fftshift(img_dft)
        img_dft_mag = np.abs(img_dft_shift)

        self.plotResult(np.log(img_dft_mag), 'Espectro em frequência')
        cv2.imwrite('./Results/{}-fourier-espectro.png'.format(self.name_file), np.log(img_dft_mag))

        img_idft = np.fft.ifft2(img_dft)
        img_inversa = np.abs(img_idft)

        self.plotResult(img_inversa, 'Imagem após IDFT')
        cv2.imwrite('./Results/{}-fourier-inversa.png'.format(self.name_file), img_inversa)

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

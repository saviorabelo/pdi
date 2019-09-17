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

    def initKernel(self):
        if self.kernel_type == 'mean':
            n = self.kernel_size
            self.kernel = (1.0/(n*n)) * np.ones((n, n))
        if self.kernel_type == 'median':
            n = self.kernel_size
            self.kernel = np.ones((n, n))
        elif self.kernel_type == 'gaussian':
            k = self.kernel_size
            aux = cv2.getGaussianKernel(ksize=k,sigma=1)
            self.kernel = aux @ aux.T
        elif self.kernel_type == 'laplacian':
            self.kernel_size = 3
            self.kernel = np.array(([0, -1, 0], 
                                    [-1, 4, -1], 
                                    [0, -1, 0]), dtype='float')
        elif self.kernel_type == 'sobelX':
            self.kernel_size = 3
            self.kernel = np.array(([-1, 0, 1], 
                                    [-2, 0, 2], 
                                    [-1, 0, 1]), dtype='float')
        elif self.kernel_type == 'sobelY':
            self.kernel_size = 3
            self.kernel = np.array(([-1, -2, -1], 
                                    [0, 0, 0], 
                                    [1, 2, 1]), dtype='float')
        elif self.kernel_type == 'prewittX':
            self.kernel_size = 3
            self.kernel = np.array(([-1, 0, 1], 
                                    [-1, 0, 1], 
                                    [-1, 0, 1]), dtype='float')
        elif self.kernel_type == 'prewittY':
            self.kernel_size = 3
            self.kernel = np.array(([-1, -1, -1], 
                                    [0, 0, 0], 
                                    [1, 1, 1]), dtype='float')
        else:
            print('Erro, kernel type is not defined!\n')

    def convolve(self, kernel_size, kernel_type, iterations=None):
        self.kernel_size = kernel_size
        self.pad = int(kernel_size/2)
        self.kernel_type = kernel_type
        if iterations == None:
            self.iterations = 1
        else:
            self.iterations = iterations
        self.initKernel()
        
        (iH, iW) = self.image.shape
        pad = self.pad

        image_aux = np.copy(self.image)
        for _ in range(self.iterations):
            # Making border for image (Padding)
            image_border = np.zeros((pad+iH+pad, pad+iW+pad))
            image_border[pad:-pad, pad:-pad] = image_aux

            output = np.zeros((iH, iW))
            for i in range(pad, iH + pad):
                for j in range(pad, iW + pad):
                    # Region of interest (roi)
                    roi = image_border[i - pad:i + pad + 1, j - pad:j + pad + 1]
                    aux = (roi * self.kernel).sum()
                    aux = np.floor(np.maximum(aux, 0))
                    output[i - pad, j - pad] = aux
            output = np.uint8(output)
            image_aux = np.copy(output)

        self.plotResult(output, self.kernel_type)
        return output
    
    def convolveMedian(self, kernel_size, kernel_type, iterations=None):
        self.kernel_size = kernel_size
        self.pad = int(kernel_size/2)
        self.kernel_type = kernel_type
        if iterations == None:
            self.iterations = 1
        else:
            self.iterations = iterations
        self.initKernel()
        
        (iH, iW) = self.image.shape
        pad = self.pad

        image_aux = np.copy(self.image)
        for _ in range(self.iterations):
            # Making border for image (Padding)
            image_border = np.zeros((pad+iH+pad, pad+iW+pad))
            image_border[pad:-pad, pad:-pad] = image_aux

            output = np.zeros((iH, iW))
            for i in range(pad, iH + pad):
                for j in range(pad, iW + pad):
                    # Region of interest (roi)
                    roi = image_border[i - pad:i + pad + 1, j - pad:j + pad + 1]
                    aux = np.median(roi * self.kernel)
                    output[i - pad, j - pad] = aux
            output = np.uint8(output)
            image_aux = np.copy(output)

        self.plotResult(output, self.kernel_type)
        return output

    def thresholdMean(self):
        media = np.mean(self.image)
        img_bin = ((self.image >= media) * 255).astype('uint8')
        
        self.plotResult(img_bin, 'Binary')
        return img_bin, media
    
    def threshold(self, T):
        image_bin = ((self.image >= T) * 255).astype('uint8')
        #self.plotResult(image_bin, 'Threshold: {}'.format(T))
        #cv2.imwrite('./Results/{}-threshold-{}.png'.format(self.name_file, T), image_bin)

    def multiThreshold(self, T1, T2):
        image_bin1 = (self.image <= T1) * 0
        image_bin2 = ((self.image > T1) & (self.image < T2)) * 128
        image_bin3 = (self.image >= T2) * 255
        image_result = (image_bin1 + image_bin2 + image_bin3).astype('uint8')

        #self.plotResult(image_result, 'Multi Threshold')
        #cv2.imwrite('./Results/{}-multiThreshold-{}-{}.png'.format(self.name_file, T1, T2), image_result)

    def otsu(self):
        threshold_otsu, image_result = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #self.threshold(threshold_otsu)
        self.plotResult(image_result, 'Otsu')
        #cv2.imwrite('./Results/image-otsu.png', image_result)
    
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

    def plotHist(self):
        hist_args = {'title':'Histogram', 
                    'xlabel':'Gray',
                    'ylabel':'Pixels',
                    'xticks':[0, 32, 64, 96, 128, 160, 192, 224, 256]}

        fig = plt.figure(figsize=(9,4.5), dpi=80)

        a = fig.add_subplot(1,2,1)
        a.imshow(self.image, cmap=plt.get_cmap('gray'))
        a.axis('off')
        a.set_title('Original')

        a = fig.add_subplot(1,2,2, **hist_args)
        a.hist(self.image.ravel(), bins=128, normed=True, edgecolor='none', histtype='stepfilled')
        plt.tight_layout()
        plt.show()

    def histogram(self):
        x = [i for i in range(1,256)]
        y = [np.sum(self.image == i) for i in x]
        
        p1 = plt.bar(x, y, 0.5)
        plt.ylabel('Grey scale')
        plt.xlabel('Pixel')
        plt.title('Histogram')
        #plt.savefig('./Results/{}-histogram.png'.format(self.name_file), bbox_inches='tight')
        plt.show()
    
    def EqHistogram(self):
        x = [i for i in range(1,256)]
        y = [np.sum(self.image == i) for i in x]
        #max_ = np.max(self.image)
        max_ = 256

        (m, n) = self.image.shape
        img = np.zeros((m, n))
        mn = m*n

        for i in range(m):
            for j in range(n):
                r = self.image[i,j]
                T = ((max_ - 1)/mn) * sum(y[0:r])
                img[i,j] = np.floor(T)

        img = np.uint8(img)
        #cv2.imwrite('./Results/{}-EqHistogram.png'.format(self.name_file), img)
        self.plotResult(img, 'EqHistogram')

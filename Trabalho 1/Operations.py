import cv2
import numpy as np
import matplotlib.pyplot as plt

class Operations:

    def __init__(self, image):
        # Check if the image has only two channels
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            self.image = image
        self.kernel = []
        self.kernel_size = 0
        self.pad = 0
        self.kernel_type = ''
        self.iterations = 0

    def initKernel(self):
        if self.kernel_type == 'mean':
            n = self.kernel_size
            self.kernel = (1.0/(n*n)) * np.ones((n, n))
        elif self.kernel_type == 'gaussian':
            k = self.kernel_size
            aux = cv2.getGaussianKernel(ksize=k,sigma=1)
            self.kernel = aux @ aux.T
        elif self.kernel_type == 'laplacian':
            self.kernel = np.array(([1, 1, 1], 
                                    [1, -8, 1], 
                                    [1, 1, 1]), dtype='float')
        elif self.kernel_type == 'sharpen':
            self.kernel = np.array(([0, -1, 0], 
                                    [-1, 5, -1], 
                                    [0, -1, 0]), dtype='float')
        elif self.kernel_type == 'sobelX':
            self.kernel = np.array(([-1, 0, 1], 
                                    [-2, 0, 2], 
                                    [-1, 0, 1]), dtype='float')
        elif self.kernel_type == 'sobelY':
            self.kernel = np.array(([-1, -2, -1], 
                                    [0, 0, 0], 
                                    [1, 2, 1]), dtype='float')
        else:
            print('Erro, kernel type is not defined!\n')

    def convolve(self, kernel_size, kernel_type, iterations):
        self.kernel_size = kernel_size
        self.pad = int(kernel_size/2)
        self.kernel_type = kernel_type
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
                    output[i - pad, j - pad] = np.floor(aux)
            output = np.uint8(output)
            #output = output.astype(np.uint16)
            image_aux = np.copy(output)

        self.plotResult(output, self.kernel_type)
        return output

    def thresholdMean(self):
        media = np.mean(self.image)
        img_bin = ((self.image > media) * 255).astype('uint8')
        
        self.plotResult(img_bin, 'Binary')
        return img_bin, media
    
    def threshold(self, T):
        img_bin = ((self.image > T) * 255).astype('uint8')
        self.plotResult(img_bin, 'Threshold: {}'.format(T))

    def otsu(self):
        threshold_otsu, image_result = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #self.threshold(threshold_otsu)
        self.plotResult(image_result, 'Otsu')
    
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

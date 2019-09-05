import cv2
import numpy as np

class Convolution:

    def __init__(self, image, kernel_size, kernel_type, iterations):
        self.image = image
        self.kernel = []
        self.kernel_size = kernel_size
        self.pad = int(kernel_size/2)
        self.kernel_type = kernel_type
        self.iterations = iterations
        self.initKernel()

    def initKernel(self):
        if self.kernel_type == 'mean':
            n = self.kernel_size
            self.kernel = (1.0/(n*n)) * np.ones((n, n))
        elif self.kernel_type == 'gaussian':
            k = self.kernel_size
            aux = cv2.getGaussianKernel(ksize=k,sigma=2)
            self.kernel = aux @ aux.T
        elif self.kernel_type == 'laplacian':
            self.kernel = np.array(([0, 1, 0], 
                                    [1, -4, 1], 
                                    [0, 1, 0]), dtype='float')
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

    def convolve(self):
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
            image_aux = np.copy(output)

        return output
    
    #def plotImage(self):

import numpy as np
import cv2

class Convolution:

    def __init__(self, image, kernel_size, kernel_type, times):
        self.image = image
        self.kernel = []
        self.kernel_size = kernel_size
        self.pad = int(kernel_size/2)
        self.kernel_type = kernel_type
        self.times = times
        self.initKernel()

    def initKernel(self):
        if self.kernel_type == 'mean':
            self.kernel = 1/9 * np.ones((self.kernel_size, self.kernel_size))

    def convolve(self):

        (iH, iW) = self.image.shape
        (kH, kW) = self.kernel.shape
        pad = self.pad

        #image = np.zeros((pad+iH+pad, pad+iW+pad))
        #image[pad:iH,pad:iW] = self.image
        
        #print(image.shape)

        image = np.copy(self.image)
        for _ in range(self.times):
            image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

            output = np.zeros((iH, iW))
            for y in np.arange(pad, iH + pad):
                for x in np.arange(pad, iW + pad):
                    roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                    aux = (roi * self.kernel).sum()
                    output[y - pad, x - pad] = np.floor(aux)
            output = output.astype(np.uint64)
            image = np.copy(output)
        
        
        return output
    
    #def plotImage(self):

# -*- coding: utf-8 -*-
# Import library
import cv2
import numpy as np
from Operations import Operations

# Load image
path = '../Images/'
name_file = 'cameraman.png'
image = cv2.imread(path+name_file)

model = Operations(image, name_file)
image_r = model.convolve(kernel_size=3, kernel_type='sobelX', iterations=1)
#cv2.imwrite('./Results/cameraman.png', image_r)

#model.otsu()
#model.threshold(150)
#model.multiThreshold(150, 170)

#gx = model.convolve(kernel_size=3, kernel_type='prewittX', iterations=1)
#gy = model.convolve(kernel_size=3, kernel_type='prewittY', iterations=1)
#g = np.sqrt(gx * gx + gy * gy)
#g *= 255.0 / np.max(g)
#g = np.uint8(g)
#cv2.imwrite('./Results/cameraman-prewitt.png', g)

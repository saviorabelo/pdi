# -*- coding: utf-8 -*-
# Import library
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Convolution import Convolution

image = cv2.imread('../Images/lena.png', cv2.IMREAD_GRAYSCALE)

model = Convolution(image=image, kernel_size=3, kernel_type='mean', times=5)
img_result = model.convolve()

plt.imshow(img_result, 'gray')
plt.show()
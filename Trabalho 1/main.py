# -*- coding: utf-8 -*-
# Import library
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Convolution import Convolution

# Load image
image = cv2.imread('../Images/lena.png', cv2.IMREAD_GRAYSCALE)

model = Convolution(image=image, kernel_size=5, kernel_type='mean', iterations=1)
img_result = model.convolve()

plt.imshow(img_result, 'gray')
plt.show()

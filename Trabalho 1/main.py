# -*- coding: utf-8 -*-
# Import library
import cv2
from Operations import Operations

# Load image
image = cv2.imread('../Images/cameraman.png')

model = Operations(image)
image_result = model.convolve(kernel_size=3, kernel_type='mean', iterations=1)
#model.otsu()

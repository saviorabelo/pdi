# -*- coding: utf-8 -*-
# Import library
import cv2
from Operations import Operations

# Load image
path = '../Images/'
name_file = 'cameraman.png'
image = cv2.imread(path+name_file)

model = Operations(image, name_file)
image_result = model.convolve(kernel_size=3, kernel_type='mean', iterations=1)
#model.otsu()
#model.threshold(150)
#model.multiThreshold(150, 170)



#sorted(p)[4]
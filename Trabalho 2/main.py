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
model.iterativeMean()


# -*- coding: utf-8 -*-
# Import library
import cv2
from Operations import Operations

# Load image
path = '../Images/'
name_file = '3303_lg.tiff'
image = cv2.imread(path+name_file)

model = Operations(image, name_file)
#model.iterativeMean()
#model.watershed()
model.regionGrowth()


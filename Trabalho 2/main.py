# -*- coding: utf-8 -*-
# Import library
import cv2
from Operations import Operations

# Load image
path = '../Images/'
name_file = 'cameraman.png'
image = cv2.imread(path+name_file)

model = Operations(image, name_file)
model.translation()
#model.scaling()
#model.rotation()
#model.se()
#model.dilation()
#model.erosion()
#model.morphologicalGradient()
#model.regionGrowth()
#model.kmeans()
#model.houghCircles()
#model.houghLines()
#model.watershed()
#model.wavelet()
#model.fourier()
#model.iterativeMean()


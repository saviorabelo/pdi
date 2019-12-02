# -*- coding: utf-8 -*-
# Import library
import cv2
import numpy as np
from utils import Data
from math import copysign, log10
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops


class Extraction:
    def huMoments():
        # Import dataset
        X, Y, lb = Data.numbers()

        X_new = []
        for img in X:

            # Convert array in image
            image = img.reshape([35, 35])
            image = image.astype(np.uint8)
            
            # Calculate Moments
            moments = cv2.moments(image)

            # Calculate Hu Moments
            huMoments = cv2.HuMoments(moments)

            # Log scale hu moments
            hm = [-1* copysign(1.0, hu) * log10(abs(hu)) if hu!=0 else 0 for hu in huMoments]

            X_new.append(hm)

        return np.array(X_new), Y, lb

    def lbp():
        # Import dataset
        X, Y, lb = Data.numbers()

        X_new = []
        for img in X:

            # Convert array in image
            image = img.reshape([35, 35])
            image = image.astype(np.uint8)
            
            # Number of points to be considered as neighbourers
            radius = 3
            no_points = 8 * radius
            # Uniform LBP is used
            lbp = local_binary_pattern(image, no_points, radius, method='uniform')
            
            lbp = np.array(lbp).flatten()
            #print(lbp)

            X_new.append(lbp)

        return np.array(X_new), Y, lb
    
    def glcm():
        # Import dataset
        X, Y, lb = Data.numbers()

        X_new = []
        for img in X:

            # Convert array in image
            image = img.reshape([35, 35])
            image = image.astype(np.uint8)
            
            g = greycomatrix(image, [0, 1], [0, np.pi/2], levels=256)

            contrast = greycoprops(g, 'contrast').flatten()
            energy = greycoprops(g, 'energy').flatten()
            homogeneity = greycoprops(g, 'homogeneity').flatten()
            correlation = greycoprops(g, 'correlation').flatten()
            dissimilarity = greycoprops(g, 'dissimilarity').flatten()
            ASM = greycoprops(g, 'ASM').flatten()

            features = np.concatenate((contrast, energy, homogeneity, correlation, dissimilarity, ASM)) 

            X_new.append(features)

        return X_new, Y, lb



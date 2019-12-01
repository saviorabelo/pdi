# -*- coding: utf-8 -*-
# Import library
import cv2
import numpy as np
from utils import Data
from math import copysign, log10


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

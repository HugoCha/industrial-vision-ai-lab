#!/usr/bin/python3

import cv2

from cv2.typing import MatLike
from enum import Enum

from src.common.processors import ImageProcessor

class ColorSpace(Enum):
    BGR=0
    HSV=1
    HLS=2
    LAB=3
    GRAY=4

class ColorNormalizer(ImageProcessor):
    def __init__( self, output_color=ColorSpace.BGR, clipLimits=2.0, tileSize=10 ):
        self.output_color = output_color
        self.clipLimits = clipLimits
        self.tileSize = tileSize
    
    def normalize( self, img:MatLike ) -> MatLike:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        l_norm = clahe.apply(l)

        lab_norm = cv2.merge([l_norm, a, b])

        if self.output_color == ColorSpace.LAB:
            return lab_norm
        
        bgr = cv2.cvtColor( lab_norm, cv2.COLOR_LAB2BGR )

        if self.output_color == ColorSpace.GRAY:
            return cv2.cvtColor( bgr, cv2.COLOR_BGR2GRAY )
        
        if self.output_color == ColorSpace.HSV:
            return cv2.cvtColor( bgr, cv2.COLOR_BGR2HSV )
        
        if self.output_color == ColorSpace.HLS:
            return cv2.cvtColor( bgr, cv2.COLOR_BGR2HLS )

        return bgr
    
    def process_img(self, img: MatLike ) -> MatLike:
        return self.normalize( img )
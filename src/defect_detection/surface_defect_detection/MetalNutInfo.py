#!/usr/bin/python3

import cv2
import numpy as np

from src.common.vision_utils import *

class HoleInfo:
    def __init__( self, center, radius ):
        self.center = center
        self.radius = radius

class MetalNutInfo:
    def __init__( self, mask, contours, hole_info ):
        self.mask = mask
        self.contours = contours
        self.hole = hole_info

    @classmethod
    def CreateFromImage( cls, img ):
        if img is None:
            raise ValueError( "Invalid img:None" )
        
        _, mask = cv2.threshold( img.astype( np.uint8 ), 35, 255, cv2.THRESH_BINARY )
        contours, _ = cv2.findContours( mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        contours = sorted( contours, key=cv2.contourArea, reverse=True )

        if ( len( contours ) > 1 ):        
            peri = cv2.arcLength( contours[1], True )
            area = cv2.contourArea( contours[1] )
            r_peri = peri / ( 2 * np.pi )
            r_area = np.sqrt( area / ( np.pi ) )
            r = ( r_peri + r_area ) / 2
            center = get_center( contours[1] )

            return cls( mask, contours, HoleInfo( center, r ) )
        
        else:
            raise ValueError( "Invalid img:No hole detected" )
        
    @property
    def main_contour( self ):
        if any( self.contours ):
            return self.contours[0]
        return []
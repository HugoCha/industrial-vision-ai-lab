#!/usr/bin/python3

import cv2
import numpy as np

from abc import ABC, abstractmethod

from src.common.vision_utils import *

from .MetalNutInfo import *
from .TemplateCreator import *

class Defect:
    def __init__( self, mask ):
        self.mask = mask
        self.contours, _ = cv2.findContours( mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

class Inspector(ABC):
    @abstractmethod
    def inspect( self, img ) -> Defect:
        pass

class InspectorType(Enum):
    ALL=0
    COLOR=1
    MATERIAL=2

class ColorDefectInspector(Inspector):
    def __init__(self, template_creator:TemplateCreator):
        self.template_creator = template_creator

    @property
    def mean( self ):
        return self.template_creator.template.mean
    
    @property
    def std( self ):
        return self.template_creator.template.std

    def inspect( self, img ) -> Defect:
        mask = np.zeros( (img.shape[1], img.shape[0]), dtype=np.uint8 )

        if self.mean is not None and self.std is not None:
            preprocess_hsv = self.template_creator.preprocess( img )
            preprocess_h, preprocess_s, preprocess_v = cv2.split( preprocess_hsv )

            mean_h, mean_s, mean_v = cv2.split( self.mean )
            std_h, std_s, std_v = cv2.split( self.std )

            _, template_mask = cv2.threshold( mean_v.astype( np.uint8 ), 50, 255, cv2.THRESH_BINARY )

            contours, _ = cv2.findContours( template_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
            contours = sorted( contours, key=cv2.contourArea, reverse=True )
            
            if ( len( contours ) >= 2 ):
                mask1 = draw_mask_contour_fill( img, contours[0] )
                mask2 = draw_mask_contour_fill( img, contours[1] )
                template_mask = cv2.bitwise_xor( mask1, mask2 )
                template_mask = cv2.drawContours( template_mask, [contours[0]], 0, 0, 4 )
                template_mask = cv2.drawContours( template_mask, [contours[1]], 0, 0, 30 )
            # blur
            delta_h = np.abs( ( preprocess_h - mean_h ) / ( std_h + 1e-6 ) )
            delta_s = np.abs( ( preprocess_s - mean_s ) / ( std_s + 1e-6 ) )

            kernel_h = np.ones((3,3), np.float32)/9.
            kernel_s = np.ones((3,3), np.float32)/9.
            mask_h = cv2.filter2D(src=delta_h, ddepth=-1, kernel=kernel_h)
            mask_s = cv2.filter2D(src=delta_s, ddepth=-1, kernel=kernel_s)
            
            mask_h = ( mask_h > 3. ).astype(np.uint8) * 255
            mask_s = ( mask_s > 3. ).astype(np.uint8) * 255

            mask = cv2.bitwise_or( mask_s, mask_h, mask=template_mask )

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

            sz = img.shape
            defect_mask = np.zeros((sz[1],sz[0]), np.uint8)

            if len( stats ) > 1:
                max_area = np.max( stats[1:, cv2.CC_STAT_AREA] )
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]

                    keepArea = area >= 9 and area < 5000 and area > max_area / 10

                    if keepArea:
                        componentMask = (labels == i).astype( np.uint8 ) * 255
                        defect_mask = cv2.bitwise_or(defect_mask, componentMask)
                
            mask = defect_mask

        return Defect( mask )

class MaterialDefectInspector(Inspector):
    def __init__(self, template_creator:TemplateCreator):
        self.template_creator = template_creator
        self.template_info = None

    @property
    def mean( self ):
        return self.template_creator.template.mean

    def inspect( self, img ) -> Defect:
        mask = np.zeros( (img.shape[1], img.shape[0]), dtype=np.uint8 )

        if self.template_info is None:
            try:
                self.template_info = MetalNutInfo.CreateFromImage( self.mean )
            except:
                return Defect( mask )

        preprocess = grayscale( img ) #self.template_creator.preprocess( img )
        preprocess = cv2.bilateralFilter( preprocess, -1, 10, 7 )
        _, img_mask = cv2.threshold( preprocess.astype( np.uint8 ), 30, 255, cv2.THRESH_BINARY )
        cv2.imshow("mask", self.template_info.mask)
        mask = cv2.bitwise_xor( self.template_info.mask, img_mask )
        mask = cv2.drawContours( mask, self.template_info.contours, 0, 0, 8 )
        cv2.circle( mask, self.template_info.hole.center, int( self.template_info.hole.radius + 10. ), 0, -1 )

        return Defect( mask )
    
class MultipleDefectInspector( Inspector ):
    def __init__( self, inspectors:Sequence[Inspector] ):
        self.inspectors = inspectors

    def inspect( self, img ) -> Defect:
        mask = np.zeros( (img.shape[1], img.shape[0]), dtype=np.uint8 )

        for inspector in self.inspectors:
            defect = inspector.inspect( img )
            mask = cv2.bitwise_or( mask, defect.mask )

        return Defect( mask )
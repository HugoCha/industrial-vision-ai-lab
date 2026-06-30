#!/usr/bin/python3

import cv2

from abc import ABC, abstractmethod

from src.common.vision_utils import *

class Defect:
    def __init__( self, mask, contours = None ):
        self.contours = contours
        self.mask = mask

class Inspector(ABC):
    def __init__(self, template_img):
        self.template_img = template_img

    @abstractmethod
    def inspect( self, img ) -> Defect:
        pass

class ShapeInspector(Inspector):
    def __init__(self, template_img):
        super().__init__(template_img)
        template_contour = get_contours(template_img)
        self.template_mask = draw_mask_contour_fill(template_img, template_contour[0])

    def inspect( self, img ) -> Defect:
        contour = get_contours(img)
        mask = draw_mask_contour_fill(img, contour[0])
        diff = cv2.absdiff( self.template_mask, mask )
        
        kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (5,5))
        morph = cv2.morphologyEx( diff, cv2.MORPH_ERODE, kernel, iterations=2 )
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph)
        
        sz = img.shape
        defect_mask = np.zeros((sz[1],sz[0]), np.uint8)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            keepArea = area > 500 and area < 1500

            if keepArea:
                componentMask = (labels == i).astype("uint8") * 255
                mask = cv2.bitwise_or(mask, componentMask)
        
        return Defect( morph )
    
class SurfaceInspector(Inspector):
    def __init__(self, template_img):
        super().__init__(template_img)

    def inspect( self, img ) -> Defect:
        return 
    
class IntensityInspector(Inspector):
    def __init__(self, template_img):
        super().__init__(template_img)

    def inspect( self, img ) -> Defect:
        return 
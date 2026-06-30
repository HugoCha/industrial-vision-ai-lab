#!/usr/bin/python3

import cv2
import numpy as np

from abc import ABC, abstractmethod
from cv2.typing import MatLike

from src.common.vision_utils import *

class TemplateMatcher(ABC):
    def __init__( self, template_img:MatLike ):
        self.template_img = template_img
        self.__template_contours = None
        self.__template_center = None
        self.__template_mask = None

    @abstractmethod
    def match( self, img_to_match: MatLike ) -> MatLike:
        return img_to_match
    
    @property
    def template_contours( self ):
        if self.__template_contours is None:
            self.__template_contours = get_contours( self.template_img )
        return self.__template_contours
    
    @property
    def template_main_contour( self ):
        if len( self.template_contours ) > 0:
            return self.template_contours[0]
        return None

    @property
    def template_center( self ):
        if ( self.template_main_contour is not None ):
            self.__template_center = get_center(self.template_main_contour)
        return self.__template_center
    
    @property
    def template_mask(self):
        if ( self.__template_mask is None ):
            if ( self.template_main_contour is not None ):
                self.__template_mask = draw_mask_contour( 
                    self.template_img, 
                    self.template_main_contour )
        return self.__template_mask

    def warp( self, img_to_match, rotation_center, angle ):
        if self.template_center is None:
            return img_to_match
        
        sz = img_to_match.shape
        translation = [self.template_center[0] - rotation_center[0], self.template_center[1] - rotation_center[1]]

        tf = self.get_affine_transform( rotation_center, angle, translation )
        rotated_image = cv2.warpAffine(
            img_to_match,
            tf,
            (sz[1],sz[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE )
        
        return rotated_image
    
    def get_affine_transform(self,rotation_center, angle, translation):
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, scale=1.0)
        affine_matrix = rotation_matrix.copy()
        affine_matrix[:, 2] += np.array([translation[0], translation[1]])
        return affine_matrix

class PCAShapeMatcher(TemplateMatcher):
    def __init__(self, template_img: MatLike ):
        super().__init__(template_img)

    def match(self, img_to_match: MatLike ) -> MatLike:
        contours = get_contours( img_to_match )
        rotated_image = img_to_match

        if len(contours) > 0:
            cnt = contours[0]
            result = get_orientation(cnt)

            if result is not None:
                center, angle, _ = result
                rotated_image = self.warp( img_to_match, center, angle )
                
        return rotated_image
    
class MinAreaRectShapeMatcher(TemplateMatcher):
    def __init__(self, template_img: MatLike ):
        super().__init__(template_img)

    def match(self, img_to_match: MatLike ) -> MatLike:
        contours = get_contours( img_to_match )
        rotated_image = img_to_match

        if len(contours) > 0:
            contour = contours[0]
            rect = cv2.minAreaRect(contour)

            center = rect[0]
            angle = rect[2]

            rotated_image = self.warp( img_to_match, center, angle )
                
        return rotated_image

class ContourFitShapeMatcher(TemplateMatcher):
    def __init__(self, template_img: MatLike ):
        super().__init__(template_img)
        if ( self.template_mask is not None ):
            self.template_distance = cv2.distanceTransform(
                255-self.template_mask,
                cv2.DIST_L2,
                5 )

    def match(self, img_to_match: MatLike ) -> MatLike:
        if ( self.template_mask is None ):
            return img_to_match
        
        contours = get_contours( img_to_match )
        rotated_image = img_to_match

        if len(contours) > 0 and self.template_center:
            contour = contours[0]
            center = get_center( contour )
            new_center = [self.template_center[0] - center[0], self.template_center[1] - center[1]]

            best_score = np.inf
            best_angle = 0
            for angle in np.arange(0, 90, 0.5):
                transform_contour = self.transform_contour( contour, center, angle, new_center )
                score = self.chamfer_distance( self.template_distance, transform_contour )
                if ( score < best_score ):
                    best_score = score
                    best_angle = angle
            
            rotated_image = self.warp( 
                img_to_match, 
                center, 
                best_angle )
            
        return rotated_image

    def transform_contour( self,
                        contour:MatLike, 
                        rotation_center, 
                        angle, 
                        new_center ) -> MatLike:
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, scale=1.0)
        rotated_contour = cv2.transform(contour.reshape(-1, 1, 2), rotation_matrix)
        rotated_contour = rotated_contour.reshape(-1, 1, 2).astype(np.int32)
        translated_contour = rotated_contour + np.array([new_center])
        return translated_contour
    
    def chamfer_distance(
        self,
        distance_image,
        contour):
        pts = contour.reshape(-1,2)
        h,w = distance_image.shape
        values = []
        for x,y in pts:
            x = int(round(x))
            y = int(round(y))

            if 0 <= x < w and 0 <= y < h:
                values.append(distance_image[y,x])

        return np.mean(values)
    
class ECCShapeMatcher(TemplateMatcher):
    def __init__(self, template_img: MatLike ):
        super().__init__(template_img)

    def match(self, img_to_match: MatLike ) -> MatLike:
        sz = self.template_img.shape
        base_gray = grayscale( self.template_img )
        img_gray = grayscale( img_to_match )

        motion_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                    1000,   # Number of iterations
                    1e-8 )  # Epsilon
        
        _, warp_matrix = cv2.findTransformECC(
            base_gray,
            img_gray,warp_matrix, 
            motion_mode, criteria)
        
        img_aligned = cv2.warpAffine(
            img_to_match, 
            warp_matrix, 
            (sz[1],sz[0]), 
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE )
 
        return img_aligned
    
class ORBShapeMatcher(TemplateMatcher):
    def __init__(self, template_img: MatLike ):
        super().__init__(template_img)

    def match(self, img_to_match: MatLike ) -> MatLike:
        if self.template_mask is None:
            return img_to_match 
        
        orb = cv2.ORB.create(1000)
        
        template_gray = self.template_mask
        gray = draw_mask_contour( img_to_match, get_contours( img_to_match )[0])
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(des1, des2, k=2)
        best_matches = []
        #sorted_matches = sorted( matches, key=lambda m: m.distance )
        #min_best_matches = min( len(matches), 20 )

        # i = 0
        # while len( best_matches ) < min_best_matches or sorted_matches[i].distance < 0.8:
        #     best_matches.append( sorted_matches[i] ) 
        #     i+=1

        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                best_matches.append([m])

        template_pts = np.float32([kp1[m[0].queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        img_pts = np.float32([kp2[m[0].trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        
        affine_matrix, inliers = cv2.estimateAffinePartial2D(
            img_pts,
            template_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0 )
        
        a, b = affine_matrix[0, :2]
        c, d = affine_matrix[1, :2]

        # Compute the scaling factor (should be 1 for rigid transformation)
        scaling_factor = np.sqrt(a**2 + c**2)

        # Normalize the rotation part to remove scaling
        affine_matrix[0, :2] /= scaling_factor
        affine_matrix[1, :2] /= scaling_factor

        # Apply the transformation to the template image
        rotated_img = cv2.warpAffine(
            img_to_match,
            affine_matrix,
            (img_to_match.shape[1], img_to_match.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE )

        return rotated_img
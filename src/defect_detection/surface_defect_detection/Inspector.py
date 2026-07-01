#!/usr/bin/python3

import cv2

from abc import ABC, abstractmethod

from src.common.vision_utils import *

class Defect:
    def __init__( self, mask ):
        self.mask = mask
        self.contours, _ = cv2.findContours( mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

class Inspector(ABC):
    def __init__(self, template_img):
        self.template_img = template_img

    @abstractmethod
    def inspect( self, img ) -> Defect:
        pass

class AddedMaterialInspector(Inspector):
    def __init__(self, template_img):
        super().__init__(template_img)
        template_contour = get_contours(template_img)
        self.template_mask = draw_mask_contour_fill(template_img, template_contour[0])

    def inspect( self, img ) -> Defect:
        contours = get_contours(img, 30)
        max_contour = get_max_contour( contours )
        mask = draw_mask_contour_fill(img, max_contour)
        diff = cv2.absdiff( self.template_mask, mask )
        kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (7,7) )
        morph = cv2.morphologyEx( diff, cv2.MORPH_OPEN, kernel )
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph)
        
        sz = img.shape
        defect_mask = np.zeros((sz[1],sz[0]), np.uint8)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            keepArea = area > 100 and area < 3000

            if keepArea:
                componentMask = (labels == i).astype("uint8") * 255
                defect_mask = cv2.bitwise_or(defect_mask, componentMask)
        
        return Defect( defect_mask )
    
class SurfaceInspector(Inspector):
    def __init__(self, template_img):
        super().__init__(template_img)
        self.template_gray = grayscale(template_img)
        #self.template_blur = cv2.GaussianBlur(self.template_gray,(19,19),0)

    def inspect( self, img ) -> Defect:
        gray = grayscale(img)
        gray = cv2.absdiff(self.template_gray,gray)
        # Global histogram equalization
        #equalized = cv2.equalizeHist(gray)

        # CLAHE (better for local contrast)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # equalized = clahe.apply(gray)
        # blur = cv2.GaussianBlur(equalized,(9,9),0)
        # Unsharp masking
        #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #sharpened = cv2.filter2D(blur, -1, kernel)

        # Laplacian filtering
        #laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        #sharpened = cv2.convertScaleAbs(laplacian)
        #thresh = cv2.adaptiveThreshold(
        #    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #    cv2.THRESH_BINARY_INV, 11, 2
        #)

        # Apply Gabor filters at multiple orientations
        gray = gray.astype(np.float32) / 255.0
        gabor_imgs = []
        for theta in np.arange(0, np.pi, np.pi/90):  # 0, 45, 90, 135 degrees
            gabor_img = self.gabor_filter(gray, theta=theta)
            gabor_imgs.append(gabor_img)

        # Combine responses (e.g., take the maximum)
        #combined_gabor = np.max(gabor_imgs, axis=0)
        combined_gabor = np.sqrt( np.sum(np.square(gabor_imgs), axis=0) )
        combined_gabor -= combined_gabor.min()
        combined_gabor /= combined_gabor.max()
        _, thresh = cv2.threshold(combined_gabor, 0.15, 1.0, cv2.THRESH_BINARY)
        thresh = (thresh * 255).astype(np.uint8)

        # diff = cv2.absdiff(self.template_blur, blur )
        # blur = cv2.GaussianBlur(gray,(19,19),0)
        # th = cv2.adaptiveThreshold( 
        #     diff, 
        #     255, 
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #     cv2.THRESH_BINARY,
        #     21,
        #     4 )
        # ksize = -1
        # gX = cv2.Sobel(diff, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
        # gY = cv2.Sobel(diff, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
        # magnitude = cv2.magnitude(gX, gY)
        # magnitude = cv2.convertScaleAbs(magnitude)

        # canny = cv2.Canny( diff, 10, 50 )
        # combine the gradient representations into a single image
        #combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
        return Defect( thresh )
    
    def gabor_filter(self, 
                     img, 
                     kernel_size=21, 
                     sigma=3, 
                     theta=0., 
                     lambda_=10, 
                     gamma=0.5, 
                     psi=0):
        # Create Gabor kernel
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), sigma, theta, lambda_, gamma, psi
        )
        kernel -= kernel.mean()
        kernel /= np.linalg.norm(kernel)
        # Apply filter
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        return filtered

class IntensityInspector(Inspector):
    def __init__(self, template_img):
        super().__init__(template_img)

    def inspect( self, img ) -> Defect:
        return 
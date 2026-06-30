#!/usr/bin/python3

import cv2
import numpy as np

from cv2.typing import MatLike

def is_grayscale( img:MatLike ) -> bool:
    return len( img.shape ) == 2

def grayscale( img:MatLike ) -> MatLike:
    return cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

def gamma_correction( img:MatLike, gamma:float=1.5 ) -> MatLike:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def otsu( img:MatLike ) -> MatLike:
    ret = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret[1]

def get_contours( img:MatLike ):
    gray = grayscale( img )
    blur = cv2.GaussianBlur(gray,(7,7),0)
    _, th = cv2.threshold( blur, 30, 255, cv2.THRESH_BINARY )
    kernel = np.ones((5,5), np.uint8)
    morphology = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = np.ones((7,7), np.uint8)
    morphology = cv2.morphologyEx(morphology, cv2.MORPH_ERODE, kernel, iterations=2)
    contours, _ = cv2.findContours(morphology, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_mask_contour_fill( img, contour ):
    sz = img.shape
    mask = np.zeros((sz[1],sz[0]), np.uint8)
    cv2.fillPoly( mask, [contour], 255 )
    return mask

def draw_mask_contour( img, contour ):
    sz = img.shape
    mask = np.zeros((sz[1],sz[0]), np.uint8)
    mask = cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        1 )
    return mask

def get_center( contour ):
    M = cv2.moments(contour)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    return (centroid_x, centroid_y)

def is_valid_contour( contour:MatLike ) -> bool:
    area = cv2.contourArea(contour)

    if area < 500 or area > 15000:
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return False

    solidity = float(area) / hull_area

    if solidity < 0.6:
        return False

    return True

def compute_circularity( contour:MatLike ) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 1e-6:
        return 0.0

    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    return circularity

def get_orientation(contour:MatLike):
    points = np.squeeze(contour).astype(np.float64)

    if len(points.shape) != 2 or len(points) < 5:
        return None

    mean, eigenvectors, _ = cv2.PCACompute2(
        points,
        mean=np.empty((0))
    )

    center = (
        int(mean[0, 0]),
        int(mean[0, 1])
    )

    vx = eigenvectors[0, 0]
    vy = eigenvectors[0, 1]

    angle = np.degrees(np.arctan2(vy, vx))

    if angle < 0:
        angle += 180

    return center, angle, (vx, vy)

def mask_background(
        img:MatLike,
        min_area:int = 1000 ) -> MatLike:
    if is_grayscale( img ):
        gray = img
    else:
        gray = grayscale( img )

    blur = cv2.GaussianBlur(gray,(7,7),0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,4 )
    kernel = np.ones((5,5), np.uint8)
    morphology = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = np.ones((7,7), np.uint8)
    morphology = cv2.morphologyEx(morphology, cv2.MORPH_ERODE, kernel, iterations=2)

    contours, _ = cv2.findContours(morphology, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    if ( is_grayscale(img) ):
        background = np.full_like(img, 255, dtype=np.uint8)
    else:
        background = np.full_like(img, (255,255,255), dtype=np.uint8)
    uniform_bg = cv2.bitwise_and(background, background, mask=mask)
    foreground = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    img_with_mask = cv2.add(uniform_bg, foreground)

    return img_with_mask

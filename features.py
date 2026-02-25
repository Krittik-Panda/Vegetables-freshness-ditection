# ==========================================
# FEATURE EXTRACTION MODULE
# ==========================================

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

def extract_features(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # ================= RGB (6) =================
    for i in range(3):
        features.append(np.mean(img[:,:,i]))
    for i in range(3):
        features.append(np.std(img[:,:,i]))

    # ================= HSV (6) =================
    for i in range(3):
        features.append(np.mean(hsv[:,:,i]))
    for i in range(3):
        features.append(np.std(hsv[:,:,i]))

    # ================= LAB (6) =================
    for i in range(3):
        features.append(np.mean(lab[:,:,i]))
    for i in range(3):
        features.append(np.std(lab[:,:,i]))

    # ================= TEXTURE (5) =================
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features.append(lap_var)

    gray_quantized = (gray / 16).astype(np.uint8)
    glcm = graycomatrix(gray_quantized,
                        distances=[1],
                        angles=[0],
                        levels=16,
                        symmetric=True,
                        normed=True)

    features.append(graycoprops(glcm, 'contrast')[0,0])
    features.append(graycoprops(glcm, 'energy')[0,0])
    features.append(graycoprops(glcm, 'homogeneity')[0,0])

    features.append(shannon_entropy(gray))

    # ================= SHAPE (6) =================
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        x,y,w,h = cv2.boundingRect(cnt)

        circularity = (4*np.pi*area)/(perimeter**2) if perimeter!=0 else 0
        solidity = area/hull_area if hull_area!=0 else 0
        aspect_ratio = w/h if h!=0 else 0
        extent = area/(w*h) if (w*h)!=0 else 0
    else:
        area=perimeter=circularity=solidity=aspect_ratio=extent=0

    features.extend([area, perimeter, circularity,
                     solidity, aspect_ratio, extent])

    # ================= DARK PIXEL RATIO (1) =================
    dark_pixels = np.sum(gray < 50)
    dark_ratio = dark_pixels / gray.size
    features.append(dark_ratio)

    return np.array(features)
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load image
img = cv.imread('coin.png')
img_copy = img.copy()

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Thresholding
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Show thresholded image
cv.imshow('Threshold', thresh)

# Noise removal (morphological opening)
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
cv.imshow('Sure Background', sure_bg)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
cv.imshow('Sure Foreground', sure_fg.astype(np.uint8))

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Mark unknown region with zero
markers[unknown == 255] = 0

# Apply watershed
markers = cv.watershed(img_copy, markers)
img_copy[markers == -1] = [255, 0, 0]  # Boundary marked in red

# Show final result
cv.imshow('Watershed Result', img_copy)

# Wait and close
cv.waitKey(0)
cv.destroyAllWindows()

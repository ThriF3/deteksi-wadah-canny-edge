import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read the image
img = cv.imread('img/coin.png')

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Otsu's thresholding (binary inverse)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("Threshold", thresh)

# Noise removal with morphological opening
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# Sure background area (dilation)
sure_bg = cv.dilate(opening, kernel, iterations=3)
cv.imshow("Sure Background", sure_bg)

# Sure foreground area using distance transform
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
cv.imshow("Sure Foreground", sure_fg)

# Finding unknown region
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labeling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that background is 1 instead of 0
markers = markers + 1

# Mark the unknown region with 0
markers[unknown == 255] = 0

# Apply watershed
markers = cv.watershed(img, markers)

# Draw boundary in red on original image
img[markers == -1] = [255, 0, 0]

# Normalize markers for display (optional)
markers_vis = cv.normalize(markers.astype('float32'), None, 0, 255, cv.NORM_MINMAX)
markers_vis = markers_vis.astype('uint8')
cv.imshow("Watershed Markers", markers_vis)

# Final result
cv.imshow("Final Result", img)

cv.waitKey(0)
cv.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('top-down1.jpg')
original = image.copy()  # Keep a copy for comparison
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Noise Removal
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Thresholding
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 3: Morphological opening
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 4: Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step 5: Sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Step 6: Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 7: Marker labeling
ret, markers = cv2.connectedComponents(sure_fg)

# Step 8: Add 1 to all labels (background is 1, not 0)
markers = markers + 1

# Step 9: Mark unknown region with 0
markers[unknown == 255] = 0

# Step 10: Apply watershed
markers = cv2.watershed(image, markers)

# Step 11: Mark boundaries on the original image
image[markers == -1] = [255, 0, 0]  # Red boundary

# Convert BGR to RGB for matplotlib
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 12: Plot using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result_rgb)
plt.title('Watershed Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()

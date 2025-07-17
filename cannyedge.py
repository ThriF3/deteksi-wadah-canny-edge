import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('img/top-down1.jpg')  # replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Preprocessing - apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=25, threshold2=200)

# Step 3: Morphological operations to close gaps in edges
kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Step 4: Find contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Draw contours and bounding boxes
result = image.copy()
circle_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 1000:
        continue

    # Fit a circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    circle_area = np.pi * radius * radius
    circle_perimeter = 2 * np.pi * radius
    contour_len = cv2.arcLength(cnt, True)

    # Ratio of contour length to full circle perimeter (how much of circle is present)
    arc_coverage = contour_len / circle_perimeter

    # Check circularity (optional refinement)
    circularity = 4 * np.pi * (area / (cv2.arcLength(cnt, True) ** 2 + 1e-5))

    # Filter based on coverage and shape
    if arc_coverage > 0.4 and 0.2 < circularity < 1.3:
        circle_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, f"#{circle_count}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

# Step 6: Display results using matplotlib
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Original
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')

# Canny Edges
axs[1].imshow(edges, cmap='gray')
# axs[1].imshow(adaptive, cmap='gray')
axs[1].set_title('Canny Edge Detection')
axs[1].axis('off')

# Detected Containers
axs[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axs[2].set_title(f'Detected Containers: {len(contours)}')
axs[2].axis('off')

plt.tight_layout()
plt.show()

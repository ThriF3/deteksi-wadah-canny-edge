import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read and preprocess image
# image = cv2.imread('img/black3.jpg')
# image = cv2.imread('img/black2.jpg')
# image = cv2.imread('img/black4.jpg')
# image = cv2.imread('img/black1.jpg')
# image = cv2.imread('img/white1.jpg')
image = cv2.imread('img/top-down1.jpg')
# image = cv2.imread('img/white2.jpg')
# image = cv2.imread('img/white4.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=0, threshold2=200)

# Make sure both images are binary (0 or 255)
thresh_bin = (thresh > 0).astype(np.uint8)
edges_bin = (edges > 0).astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=2)
dilated_edges_bin = (dilated_edges > 0).astype(np.uint8)

# XOR-like logic (invert where both are white)
# (1 if one is white and the other is black, else 0)
combined_mask = cv2.bitwise_xor(thresh_bin, dilated_edges_bin) * 255

# Optional: Clean small dots
combined_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Closing to fill small holes
closed = cv2.morphologyEx(combined_cleaned, cv2.MORPH_CLOSE, kernel)

# Find contours
# contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(dilated_edges_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Draw results
output = image.copy()
# for i, cnt in enumerate(contours):
#     if cv2.contourArea(cnt) > 1000:
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(output, f'#{i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

valid_count = 0
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        continue  # avoid division by zero

    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    # Only keep contours with area > 100 and circularity close to 1 (ideal circle)
    if area > 1000 and 0.5 < circularity < 1.2:
        valid_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, f'#{valid_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        # (x, y), radius = cv2.minEnclosingCircle(cnt)
        # center = (int(x), int(y))
        # radius = int(radius)
        # cv2.circle(output, center, radius, (0, 255, 0), 2)
        # cv2.putText(output, f'#{i+1}', (center[0] - 10, center[1] - radius - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

# Show all key stages
fig, axs = plt.subplots(2, 4, figsize=(22, 10))

axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(thresh, cmap='gray')
axs[0, 1].set_title('Thresh Binary + Otsu')
axs[0, 1].axis('off')

axs[0, 2].imshow(edges, cmap='gray')
axs[0, 2].set_title('Canny Edges')
axs[0, 2].axis('off')

axs[0, 3].imshow(dilated_edges_bin * 255, cmap='gray')
axs[0, 3].set_title('Dilated Edges')
axs[0, 3].axis('off')

axs[1, 0].imshow(combined_mask, cmap='gray')
axs[1, 0].set_title('XOR Mask')
axs[1, 0].axis('off')

# axs[1, 1].imshow(combined_cleaned, cmap='gray')
# axs[1, 1].set_title('Cleaned Mask')
axs[1, 1].axis('off')

# axs[1, 2].imshow(closed, cmap='gray')
# axs[1, 2].set_title('Closed Final Mask')
axs[1, 2].axis('off')

axs[1, 3].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axs[1, 3].set_title(f'Detected Containers: {valid_count}')
axs[1, 3].axis('off')

# Hide last empty subplot
# axs[1, 3].axis('off')

plt.tight_layout()
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

def boundingBox(contours, img, text):
    valid_count = 0
    out = img.copy()
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue  # avoid division by zero

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # Only keep contours with area > 1000 and circularity close to 1 (ideal circle)
        if area > 1000 and 0.6 < circularity < 1.2:
            valid_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(out, f'#{valid_count}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    return out, valid_count, text

# Load and preprocess image
# image = cv2.imread('img/top-down1.jpg')
# image = cv2.imread('img/black3.jpg')
# image = cv2.imread('img/black2.jpg')
# image = cv2.imread('img/black4.jpg')
# image = cv2.imread('img/black1.jpg')
# image = cv2.imread('img/white1.jpg')
image = cv2.imread('img/white2.jpg')
# image = cv2.imread('img/white4.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Canny Edge Detection
edges = cv2.Canny(blurred, 0, 200)

# Convert to binary masks
thresh_bin = (thresh > 0).astype(np.uint8)
edges_bin = (edges > 0).astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=2)
dilated_edges_bin = (dilated_edges > 0).astype(np.uint8)

# XOR-like logic
combined_mask = cv2.bitwise_xor(thresh_bin, dilated_edges_bin) * 255
invert_xor = cv2.bitwise_not(combined_mask)

# Optional: Clean small dots
combined_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Closing to fill small holes
closed = cv2.morphologyEx(combined_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find contours on final mask (e.g., dilated_edges_bin or combined_mask)
contours_XOR, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_dilate, _ = cv2.findContours(dilated_edges_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_canny, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_open, _ = cv2.findContours(combined_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_close, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_invertXOR, _ = cv2.findContours(invert_xor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Call the function and get output image
output_XOR, count_XOR, text_XOR = boundingBox(contours_XOR, image, 'XOR')
output_dilate, count_dilate, text_dilate = boundingBox(contours_dilate, image, 'Dilate')
output_canny, count_canny, text_canny = boundingBox(contours_canny, image, 'Canny')
output_thresh, count_thresh, text_thresh = boundingBox(contours_thresh, image, 'Thresh')

output_open, count_open, text_open = boundingBox(contours_open, image, 'Opening')
output_close, count_close, text_close = boundingBox(contours_close, image, 'Closing')

output_invertXOR, count_invertXOR, text_invertXOR = boundingBox(contours_invertXOR, image, 'Invert XOR')

# # Show all key stages
# fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# axs[0, 0].imshow(thresh, cmap='gray')
# axs[0, 0].set_title('Thresh Binary + Otsu')
# axs[0, 0].axis('off')

# axs[0, 1].imshow(edges, cmap='gray')
# axs[0, 1].set_title('Canny Edges')
# axs[0, 1].axis('off')

# axs[0, 2].imshow(dilated_edges_bin * 255, cmap='gray')
# axs[0, 2].set_title('Dilated Edges')
# axs[0, 2].axis('off')

# axs[0, 3].imshow(combined_mask, cmap='gray')
# axs[0, 3].set_title('XOR Mask')
# axs[0, 3].axis('off')

# axs[1, 0].imshow(cv2.cvtColor(output_thresh, cv2.COLOR_BGR2RGB))
# axs[1, 0].set_title(f'Detected Containers ({text_thresh}): {count_thresh}')
# axs[1, 0].axis('off')

# axs[1, 1].imshow(cv2.cvtColor(output_canny, cv2.COLOR_BGR2RGB))
# axs[1, 1].set_title(f'Detected Containers ({text_canny}): {count_canny}')
# axs[1, 1].axis('off')

# axs[1, 2].imshow(cv2.cvtColor(output_dilate, cv2.COLOR_BGR2RGB))
# axs[1, 2].set_title(f'Detected Containers ({text_dilate}): {count_dilate}')
# axs[1, 2].axis('off')

# axs[1, 3].imshow(cv2.cvtColor(output_XOR, cv2.COLOR_BGR2RGB))
# axs[1, 3].set_title(f'Detected Containers ({text_XOR}): {count_XOR}')
# axs[1, 3].axis('off')

# plt.tight_layout()
# plt.show()
# plt.clf()

fig, axs = plt.subplots(2, 2, figsize=(20, 10))

axs[0, 0].imshow(combined_mask, cmap='gray')
axs[0, 0].set_title('XOR Mask')
axs[0, 0].axis('off')

axs[0, 1].imshow(invert_xor, cmap='gray')
axs[0, 1].set_title('Invert XOR Mask')
axs[0, 1].axis('off')

axs[1, 0].imshow(cv2.cvtColor(output_XOR, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title(f'Detected Containers ({text_XOR}): {count_XOR}')
axs[1, 0].axis('off')

axs[1, 1].imshow(cv2.cvtColor(output_invertXOR, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title(f'Detected Containers ({text_invertXOR}): {count_invertXOR}')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
plt.clf()

blended = cv2.addWeighted(output_XOR, 0.5, output_invertXOR, 0.5, 0)
fig, axs = plt.subplots(1, 1, figsize=(20, 10))
axs.imshow(blended, cmap='gray')
axs.set_title('Blended')
axs.axis('off')

plt.tight_layout()
plt.show()

# axs[1, 0].imshow(combined_cleaned, cmap='gray')
# axs[1, 0].set_title('Morph Open')
# axs[1, 0].axis('off')
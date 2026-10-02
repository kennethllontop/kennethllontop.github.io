import numpy as np
from scipy.signal import convolve2d # Built-in function
import cv2 # Using opencv for the grayscale images

# Process img as grayscale
pic = 'cameraman.png'
img = cv2.imread(f'pictures/{pic}', 0)
img = np.asarray(img, dtype=np.float64)

# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Write out difference operators
diff_x = np.array([[1, 0, -1]])
diff_y = np.array([[1], [0], [-1]])

# convolve the picture with the difference operators
result_diff_x = convolve2d(img, diff_x, mode='same')
result_diff_y = convolve2d(img, diff_y, mode='same')

partial_dx = (result_diff_x + 255) /2
partial_dy = (result_diff_y + 255) /2

cv2.imwrite('output_pictures/diff_x_cameraman.png', partial_dx.astype(np.uint8))
cv2.imwrite('output_pictures/diff_y_cameraman.png', partial_dy.astype(np.uint8))

# Calculating the magnitude
magnitude = np.sqrt((result_diff_x)**2 + (result_diff_y)**2)
mag_img = magnitude / magnitude.max() * 255 # Rescaling
cv2.imwrite('output_pictures/magnitude_cameraman.png', mag_img.astype(np.uint8))

edges_mask = magnitude > 65
cv2.imwrite('output_pictures/magnitude_cameraman_binary.png', edges_mask.astype(np.uint8) * 255)

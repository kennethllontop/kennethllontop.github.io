# DoG Filter
# Create a blurred version of the original image by convolving with a gaussian
# Show the gradient magnitude image and binarized gradient magnitude image. 
import numpy as np
from scipy.signal import convolve2d # Built-in function
import cv2 # Using opencv for the grayscale images

def gaussian_filter(img, kernel): # Copying code from project 1
    blurry_img = convolve2d(img, kernel, mode="same", boundary ="symm") # mode is same to keep the same dim; boundary is symm to not average edges to black
    return blurry_img

# Create Kernel
kernel = cv2.getGaussianKernel(ksize=3, sigma=0.6)
kernel_2d = kernel @ kernel.T

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

# convolve the blurry picture with the difference operators
blurry_image = gaussian_filter(img, kernel_2d)

result_diff_x = convolve2d(blurry_image, diff_x, mode='same')
result_diff_y = convolve2d(blurry_image, diff_y, mode='same')

partial_dx = (result_diff_x + 255) /2
partial_dy = (result_diff_y + 255) /2

cv2.imwrite('output_pictures/diff_x_cameraman_blurry.png', partial_dx.astype(np.uint8))
cv2.imwrite('output_pictures/diff_y_camerama_blurry.png', partial_dy.astype(np.uint8))

# Calculating the magnitude
magnitude = np.sqrt((result_diff_x)**2 + (result_diff_y)**2)
mag_img = magnitude / magnitude.max() * 255 # Rescaling
cv2.imwrite('output_pictures/magnitude_cameraman_blurry.png', mag_img.astype(np.uint8))

edges_mask = magnitude > 65
cv2.imwrite('output_pictures/magnitude_cameraman_binary_blurry.png', edges_mask.astype(np.uint8) * 255)
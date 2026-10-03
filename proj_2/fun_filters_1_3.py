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

# Binary
edges_mask = magnitude > 65
cv2.imwrite('output_pictures/magnitude_cameraman_binary_blurry.png', edges_mask.astype(np.uint8) * 255)

# DoG Filter
dog_x = convolve2d(kernel_2d, diff_x, mode='full')
dog_y = convolve2d(kernel_2d, diff_y, mode='full')

#Applying DoG Filter
result_dog_x = convolve2d(img, diff_x, mode='same')
result_dog_y = convolve2d(img, diff_y, mode='same')


cv2.imwrite('output_pictures/dog_x_cameraman.png', result_dog_x.astype(np.uint8))
cv2.imwrite('output_pictures/dog_y_cameraman.png', result_dog_y.astype(np.uint8))

# Calculating the magnitude
magnitude_dog = np.sqrt((result_dog_x)**2 + (result_dog_y)**2)
mag_img_dog = magnitude_dog / magnitude_dog.max() * 255 # Rescaling
cv2.imwrite('output_pictures/magnitude_cameraman_dog.png', mag_img_dog.astype(np.uint8))

# Binary
edges_mask_dog = magnitude_dog > 65
cv2.imwrite('output_pictures/magnitude_cameraman_binary_dog.png', edges_mask_dog.astype(np.uint8) * 255)


# DoG Filter
# Create a blurred version of the original image by convolving with a gaussian
# Show the gradient magnitude image and binarized gradient magnitude image. 
import numpy as np
from scipy.signal import convolve2d # Built-in function
import cv2 # Using opencv for the grayscale images
import matplotlib.pyplot as plt

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

# Showing the filters as images
filters = [kernel_2d, dog_x, dog_y]
filter_titles = ['Gaussian', 'DoG x', 'DoG y']

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i in range(0, len(filters)):
    filt = axes[i].imshow(filters[i], cmap='gray', interpolation='nearest') # nearest so each value shows as one block
    axes[i].set_title(filter_titles[i])
    axes[i].axis('off')
    fig.colorbar(filt, ax=axes[i]) # Shows which values are negative and positive
plt.tight_layout()
plt.savefig('output_pictures/dog_filters.png', dpi=150)
plt.close()

#Applying DoG Filter
result_dog_x = convolve2d(img, dog_x, mode='same')
result_dog_y = convolve2d(img, dog_y, mode='same')


partial_dog_x = (result_dog_x + 255) /2
partial_dog_y = (result_dog_y + 255) /2

cv2.imwrite('output_pictures/dog_x_cameraman.png', partial_dog_x.astype(np.uint8))
cv2.imwrite('output_pictures/dog_y_cameraman.png', partial_dog_y.astype(np.uint8))

# Calculating the magnitude
magnitude_dog = np.sqrt((result_dog_x)**2 + (result_dog_y)**2)
mag_img_dog = magnitude_dog / magnitude_dog.max() * 255 # Rescaling
cv2.imwrite('output_pictures/magnitude_cameraman_dog.png', mag_img_dog.astype(np.uint8))

# Binary
edges_mask_dog = magnitude_dog > 65
cv2.imwrite('output_pictures/magnitude_cameraman_binary_dog.png', edges_mask_dog.astype(np.uint8) * 255)

# Verifying that blur then difference gives the same result as the DoG filter
# The blur uses boundary="symm" but the difference operators pad with zeros, so the outer 2 pixels will not match
b = 2
print("Max difference in x (ignoring the border):", np.abs(result_diff_x - result_dog_x)[b:-b, b:-b].max())
print("Max difference in y (ignoring the border):", np.abs(result_diff_y - result_dog_y)[b:-b, b:-b].max())
print("Same result:", np.allclose(result_diff_x[b:-b, b:-b], result_dog_x[b:-b, b:-b]) and
                      np.allclose(result_diff_y[b:-b, b:-b], result_dog_y[b:-b, b:-b]))


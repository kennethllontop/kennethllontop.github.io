import matplotlib.pyplot as plt
from align_image_code import align_images
from scipy.signal import convolve2d # Built-in function
import cv2 # Using for gaussian kernel
import numpy as np

# First load images

# # high sf
# im2 = plt.imread('./DerekPicture.jpg') / 255.

# # low sf
# im1 = plt.imread('./nutmeg.jpg') / 255.

# high sf
im2 = plt.imread('./empty.jpg') / 255.

# low sf
im1 = plt.imread('./soup.jpg') / 255.

# # # high sf
# im1 = plt.imread('./stitch.jpg') / 255.

# # low sf
# im2 = plt.imread('./toothless.jpg') / 255.

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

def hybrid_image(im1, im2, sigma1, sigma2):
    # Img 1

    # Create Gaussian Kernel
    k = int(np.ceil(6 * sigma1)) | 1
    g_kernel = cv2.getGaussianKernel(ksize=k, sigma=sigma1)
    g_kernel_2d = g_kernel @ g_kernel.T

    #Getting the individual channels
    red = im1[:, :, 0]
    green = im1[:, :, 1]
    blue = im1[:, :, 2]

    blurry_red = convolve2d(red, g_kernel_2d, mode="same", boundary ="symm")
    blurry_green = convolve2d(green, g_kernel_2d, mode="same", boundary ="symm")
    blurry_blue = convolve2d(blue, g_kernel_2d, mode="same", boundary ="symm")

    blurry = np.dstack([blurry_red, blurry_green, blurry_blue])

    # high pass img
    hp_img = im1 - blurry

    # Img 2

    # Create Gaussian Kernel
    k = int(np.ceil(6 * sigma2)) | 1
    g_kernel = cv2.getGaussianKernel(ksize=k, sigma=sigma2)
    g_kernel_2d = g_kernel @ g_kernel.T

    #Getting the individual channels
    red = im2[:, :, 0]
    green = im2[:, :, 1]
    blue = im2[:, :, 2]

    blurry_red = convolve2d(red, g_kernel_2d, mode="same", boundary ="symm")
    blurry_green = convolve2d(green, g_kernel_2d, mode="same", boundary ="symm")
    blurry_blue = convolve2d(blue, g_kernel_2d, mode="same", boundary ="symm")

    lp_img = np.dstack([blurry_red, blurry_green, blurry_blue])

    combination = np.clip(lp_img + hp_img, 0, 1)

    return combination, hp_img, lp_img


# Converts a color image to grayscale w/ the standard luminance weights
def to_gray(im):
    red = im[:, :, 0]
    green = im[:, :, 1]
    blue = im[:, :, 2]
 
    gray = 0.2125 * red + 0.7154 * green + 0.0721 * blue
    return gray
 
 
# Log magnitude of the Fourier transform (with a small epsilon)
def log_fft(gray_image):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image))) + 1e-8)
 
 
sigma1 = 2
sigma2 = 4
hybrid, hp_img, lp_img = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
 
plt.imshow(hybrid)
plt.imsave('./output_pictures/soup_hybrid.jpg', hybrid)
plt.show()
 
# # Frequency analysis (only needed for my favorite result)
 
# fft_images = [im1_aligned, im2_aligned, hp_img, lp_img, hybrid]
# fft_titles = ['Input 1 (high sf)', 'Input 2 (low sf)', 'High pass', 'Low pass', 'Hybrid']
 
# fig, axes = plt.subplots(1, 5, figsize=(20, 4))
# for i in range(0, len(fft_images)):
#     gray = to_gray(fft_images[i])
#     axes[i].imshow(log_fft(gray), cmap='gray')
#     axes[i].set_title(fft_titles[i])
#     axes[i].axis('off')
 
# plt.tight_layout()
# plt.savefig('./output_pictures/toothless_fft_analysis.png', dpi=150)
# plt.show()
 
# Grayscale hybrid (optional, to compare with the color version)
 
gray1 = to_gray(im1_aligned)
gray2 = to_gray(im2_aligned)
 
# hybrid_image expects 3 channels, so stack the gray image into 3 identical ones
gray1_3ch = np.dstack([gray1, gray1, gray1])
gray2_3ch = np.dstack([gray2, gray2, gray2])
 
hybrid_gray, _, _ = hybrid_image(gray1_3ch, gray2_3ch, sigma1, sigma2)
hybrid_gray = hybrid_gray[:, :, 0] # all 3 channels are the same, so keep one
 
plt.imshow(hybrid_gray, cmap='gray')
plt.imsave('./output_pictures/soup_hybrid_gray.jpg', hybrid_gray, cmap='gray')
plt.show()
import matplotlib.pyplot as plt
from align_image_code import align_images
from scipy.signal import convolve2d # Built-in function
import cv2 # Using for gaussian kernel
import numpy as np

# First load images

# main.py passes the files and names for each pair, otherwise these defaults are used
# low sf
im2 = plt.imread(globals().get('low_file', './DerekPicture.jpg')) / 255.

# high sf
im1 = plt.imread(globals().get('high_file', './nutmeg.jpg')) / 255.

high_name = globals().get('high_name', 'nutmeg')
low_name = globals().get('low_name', 'derek')
hybrid_name = globals().get('hybrid_name', 'cat_derek_hybrid')

# # low sf
# im2 = plt.imread('./empty.jpg') / 255.

# # high sf
# im1 = plt.imread('./soup.jpg') / 255.

# # high sf
# im1 = plt.imread('./stitch.jpg') / 255.

# # low sf
# im2 = plt.imread('./toothless.jpg') / 255.

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

# I am going to save the aligned images
plt.imsave(f'./output_pictures/{high_name}_aligned.jpg', im1_aligned)
plt.imsave(f'./output_pictures/{low_name}_aligned.jpg', im2_aligned)

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
 
 
sigma1 = globals().get('sigma1', 10)
sigma2 = globals().get('sigma2', 15) # The finetuning of sigma1 and sigma2 was mostly done to get better results on my images instead of the derek and nutmeg.
# I chose a relatively small sigma1 because I wanted to keep only Stitches fine lines. I chose sigma2 to put a stronger blur on toothless, so I can focus on stitch up close.
# For derek I chose sigma1 10 and sigma2 15, for my other images I used sigma1 2 and sigma2 6
hybrid, hp_img, lp_img = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
 
plt.imshow(hybrid)
plt.imsave(f'./output_pictures/{hybrid_name}.jpg', hybrid)
plt.show()

# Saving the high pass image and low pass image
plt.imsave(f'./output_pictures/{low_name}_lp.jpg', lp_img)
plt.imsave(f'./output_pictures/{high_name}_hp.jpg', np.clip(hp_img + 0.5, 0, 1))
 
# Frequency analysis (only needed for my favorite result)
make_fft = globals().get('make_fft', False)
if make_fft:
    fft_images = [im1_aligned, im2_aligned, hp_img, lp_img, hybrid]
    fft_titles = ['Input 1 (high sf)', 'Input 2 (low sf)', 'High pass', 'Low pass', 'Hybrid']

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(0, len(fft_images)):
        gray = to_gray(fft_images[i])
        axes[i].imshow(log_fft(gray), cmap='gray')
        axes[i].set_title(fft_titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'./output_pictures/{low_name}_fft_analysis.png', dpi=150)
    plt.show()
 
# Grayscale hybrid (optional, to compare with the color version)
 
gray1 = to_gray(im1_aligned)
gray2 = to_gray(im2_aligned)
 
# hybrid_image expects 3 channels, so stack the gray image into 3 identical ones
gray1_3ch = np.dstack([gray1, gray1, gray1])
gray2_3ch = np.dstack([gray2, gray2, gray2])
 
hybrid_gray, _, _ = hybrid_image(gray1_3ch, gray2_3ch, sigma1, sigma2)
hybrid_gray = hybrid_gray[:, :, 0] # all 3 channels are the same, so keep one
 
plt.imshow(hybrid_gray, cmap='gray')
plt.imsave(f'./output_pictures/{hybrid_name}_gray.jpg', hybrid_gray, cmap='gray')
plt.show()
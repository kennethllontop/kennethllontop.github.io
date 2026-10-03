# Image Sharpening

import numpy as np
from scipy.signal import convolve2d # Built-in function
import cv2 # Using opencv for the grayscale images


# I need to blur the original image. Subtract the blur from the image and we are left with the high frequencies. We add the high frequencies to the image
# We need to combine this into one convolution

# Create Gaussian Kernel
k = 5
g_kernel = cv2.getGaussianKernel(ksize=k, sigma=0.8)
g_kernel_2d = g_kernel @ g_kernel.T

# Create Identity Kernel
identity_kernel = np.zeros((k, k), dtype=np.float32)
identity_kernel[k//2, k//2] = 1.0

# # Controls how strong the sharpening is
# alpha = 1

# # Unsharp Mask Filter
# unsharp = (1+alpha) * identity_kernel - alpha * g_kernel_2d

# Process img in color
pictures = ['taj', 'taj', 'taj', 'taj', "mexico_kenneth_edith", 'family', 'house']
for i in range(len(pictures)):
    pic = pictures[i]
    if pic == 'taj':
        alpha = (2)**i
        unsharp = (1+alpha) * identity_kernel - alpha * g_kernel_2d
    else:
        alpha = 4
        unsharp = (1+alpha) * identity_kernel - alpha * g_kernel_2d

    img = cv2.imread(f'pictures/{pic}.jpg')
    img = np.asarray(img, dtype=np.float64)

    #Getting the individual channels
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    # Getting the blur and high frequencies
    blur_blue = convolve2d(blue, g_kernel_2d, mode="same", boundary ="symm")
    blur_green = convolve2d(green, g_kernel_2d, mode="same", boundary ="symm")
    blur_red = convolve2d(red, g_kernel_2d, mode="same", boundary ="symm")
    blurry_img = np.dstack([blur_blue, blur_green, blur_red])

    # Blurry Image
    cv2.imwrite(f'output_pictures/fig_2_1/blur_{pic}.png', blurry_img.astype(np.uint8))

    # High Frequencies
    cv2.imwrite(f'output_pictures/fig_2_1/hf_{pic}.png', (img - blurry_img).astype(np.uint8) + 128)


    sharp_blue = convolve2d(blue, unsharp, mode="same", boundary ="symm")
    sharp_green = convolve2d(green, unsharp, mode="same", boundary ="symm")
    sharp_red = convolve2d(red, unsharp, mode="same", boundary ="symm")
    sharp_img = np.dstack([sharp_blue, sharp_green, sharp_red])

    cv2.imwrite(f'output_pictures/fig_2_1/{pic}_alpha{str(alpha)}.png', np.clip(sharp_img, 0, 255).astype(np.uint8)) # Clip because pushes images above 255

# Evaluation: Sharp Image --> Blur --> Sharpen

# Process Image
pic = 'family_v2'
img = cv2.imread(f'pictures/{pic}.jpg')
img = np.asarray(img, dtype=np.float64)

# Controls how strong the sharpening is
alpha = 4

# Unsharp Mask Filter
unsharp = (1+alpha) * identity_kernel - alpha * g_kernel_2d

#Getting the individual channels
blue = img[:, :, 0]
green = img[:, :, 1]
red = img[:, :, 2]

blur_blue = convolve2d(blue, g_kernel_2d, mode="same", boundary ="symm")
blur_green = convolve2d(green, g_kernel_2d, mode="same", boundary ="symm")
blur_red = convolve2d(red, g_kernel_2d, mode="same", boundary ="symm")
blurry_img = np.dstack([blur_blue, blur_green, blur_red])

# Blurry Image
cv2.imwrite(f'output_pictures/fig_2_1/blur_{pic}.png', blurry_img.astype(np.uint8))

# Getting the individual channels
blue = blurry_img[:, :, 0]
green = blurry_img[:, :, 1]
red = blurry_img[:, :, 2]


sharp_blue = convolve2d(blue, unsharp, mode="same", boundary ="symm")
sharp_green = convolve2d(green, unsharp, mode="same", boundary ="symm")
sharp_red = convolve2d(red, unsharp, mode="same", boundary ="symm")
sharp_img = np.dstack([sharp_blue, sharp_green, sharp_red])

cv2.imwrite(f'output_pictures/fig_2_1/{pic}.png', np.clip(sharp_img, 0, 255).astype(np.uint8)) # Clip because pushes images above 255

# Gaussian and Laplacian Stacks
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as skio
import cv2

# Added this import so I can convolve
from scipy import signal

def gaussian_filter(img, sigma):
    k = int(np.ceil(6 * sigma)) | 1
    g_kernel = cv2.getGaussianKernel(ksize=k, sigma=sigma)
    g_kernel_2d = g_kernel @ g_kernel.T
    blurry_img = signal.convolve2d(img, g_kernel_2d, mode="same", boundary ="symm") # mode is same to keep the same dim; boundary is symm to not average edges to black
    return blurry_img

def gaussian_stack(im, sig):
    
    new_r = im[:,:,0]
    new_g = im[:,:,1]
    new_b = im[:,:,2]

    blurry_b = gaussian_filter(new_b, sigma=sig)
    blurry_g = gaussian_filter(new_g, sigma=sig)
    blurry_r =  gaussian_filter(new_r, sigma=sig)

    rgb = np.dstack([blurry_r, blurry_g, blurry_b])

    return rgb
    
s = 2
levels = 7
# pictures = ['apple', 'orange']

# pictures = ['night', 'ocean']

pictures = ['blue_shell', 'water']
g_stack = []
l_stack = []
for pic in pictures:
    imname = f'./pictures/{pic}.jpeg'
    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)

    # This code will be the whole blur
    py_images = [im]

    current_img = im

    for i in np.arange(1, levels): # already initialized it with the big image
        # I will need to blur b, g, r
        sig = s * 2 ** (i - 1) # I need this to increase the blur per level without downsampling

        imout = gaussian_stack(current_img, sig)
        py_images.append(imout)
        current_img = imout
    
    lap_stack = []
    for i in range(0, len(py_images) - 1):
        lap_stack.append(py_images[i] - py_images[i+1])
    lap_stack.append(py_images[-1])
    
    l_stack.append(lap_stack)
    g_stack.append(py_images)

# G_stack and L_stack have the images at eahc level for each image
# Getting the gaussian stack of the mask

height, width = g_stack[0][0].shape[:2]
mask = np.zeros((height, width, 3))

# Vertical Mask
# mask[:, :width // 2] = 1.0

# Horizontal Mask
# mask[:height // 2, :] = 1.0

# Human mask
mask = skio.imread('./pictures/person_mask.png') / 255.0
mask = np.dstack([mask, mask, mask])

mask = gaussian_stack(mask, 1)
mask_stack_images = [mask]

current_mask = mask

for i in np.arange(1, levels): # already initialized it with the big image
    # I will need to blur b, g, r
    sig = s * 2 ** (i - 1) # I need this to increase the blur per level without downsampling
    maskout = gaussian_stack(current_mask, sig)
    mask_stack_images.append(maskout)
    current_mask = maskout

LA, LB = l_stack[0], l_stack[1]
layer_collections = []

for i in range(0, levels):
    layer = mask_stack_images[i] * LA[i] + (1- mask_stack_images[i]) * LB[i]
    layer_collections.append(layer)

result = np.clip(sum(layer_collections), 0, 1)
skio.imsave('output_pictures/cold_water.jpg', (result * 255).astype(np.uint8))
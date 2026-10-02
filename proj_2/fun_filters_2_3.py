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

s = 2
pictures = ['apple', 'orange']
g_stack = []
l_stack = []
for pic in pictures:
    imname = f'./pictures/{pic}.jpeg'
    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
    
    new_r = im[:,:,0]
    new_g = im[:,:,1]
    new_b = im[:,:,2]

    # This code will be the whole blur
    py_lvls = 5
    py_images = [im]

    for i in np.arange(1, py_lvls): # already initialized it with the big image
        # I will need to blur b, g, r
        sig = s * 2 ** (i - 1) # I need this to increase the blur per level without downsampling
        blurry_b = gaussian_filter(new_b, sigma=sig)
        blurry_g = gaussian_filter(new_g, sigma=sig)
        blurry_r =  gaussian_filter(new_r, sigma=sig)

        rgb = np.dstack([blurry_r, blurry_g, blurry_b])

        py_images.append(rgb)

        new_b = blurry_b
        new_g = blurry_g
        new_r = blurry_r
    
    lap_stack = []
    for i in range(0, len(py_images) - 1):
        lap_stack.append(py_images[i] - py_images[i+1])
    lap_stack.append(py_images[-1])
    
    l_stack.append(lap_stack)
    g_stack.append(py_images)


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

for pic, g_levels, l_levels in zip(pictures, g_stack, l_stack):
    fig, axes = plt.subplots(2, len(g_levels), figsize=(3 * len(g_levels), 6))
    for i in range(len(g_levels)):
        axes[0, i].imshow(np.clip(g_levels[i], 0, 1))
        l = l_levels[i] if i == len(l_levels) - 1 else normalize(l_levels[i])
        axes[1, i].imshow(np.clip(l, 0, 1))
        axes[0, i].set_title(f'Level {i}')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(f'output_pictures/{pic}_stacks.png', dpi=150)
    plt.close()
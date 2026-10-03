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
pictures = ['apple', 'orange']

# pictures = ['night', 'ocean']

# pictures = ['blue_shell', 'water']
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
mask[:, :width // 2] = 1.0

# Horizontal Mask
# mask[:height // 2, :] = 1.0

# # Human mask
# mask = skio.imread('./pictures/person_mask.png') / 255.0
# mask = np.dstack([mask, mask, mask])

# Mask code
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
skio.imsave('output_pictures/oraple.jpg', (result * 255).astype(np.uint8))


# Creating the figure
def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

# Masking the laplacian levels of each image
masked_A = []
masked_B = []
for i in range(0, levels):
    masked_A.append(mask_stack_images[i] * LA[i])
    masked_B.append((1 - mask_stack_images[i]) * LB[i])

# Levels i want to show
show_levels = [0, 2, 4]

fig, axes = plt.subplots(len(show_levels) + 1, 3, figsize=(9, 3 * (len(show_levels) + 1)))
for row in range(0, len(show_levels)):
    i = show_levels[row]
    # Laplacian levels have negative values so I need to normalize them to see them
    axes[row, 0].imshow(normalize(masked_A[i]))
    axes[row, 1].imshow(normalize(masked_B[i]))
    axes[row, 2].imshow(normalize(layer_collections[i]))
    axes[row, 0].set_title(f'{pictures[0]} Level {i}')
    axes[row, 1].set_title(f'{pictures[1]} Level {i}')
    axes[row, 2].set_title(f'Blended Level {i}')

# Last row is every level added together
axes[-1, 0].imshow(np.clip(sum(masked_A), 0, 1))
axes[-1, 1].imshow(np.clip(sum(masked_B), 0, 1))
axes[-1, 2].imshow(np.clip(sum(layer_collections), 0, 1))
axes[-1, 0].set_title(f'{pictures[0]} All Levels')
axes[-1, 1].set_title(f'{pictures[1]} All Levels')
axes[-1, 2].set_title('Blended All Levels')

for row in range(0, len(show_levels) + 1):
    for col in range(0, 3):
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('output_pictures/oraple_laplacian.png', dpi=150)
plt.close()
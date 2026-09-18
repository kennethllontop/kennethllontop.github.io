# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

# name of the input file
pic = 'tobolsk.jpg'
imname = f'CS180_fa2026_proj1_data/{pic}'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
print('Image shape', im.shape)
height = np.floor(im.shape[0] / 3.0).astype(np.int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# We first want to align g to b and then r to b
# Keep track of the vector that aligns g to b and r to b; we will need to print it out
# Assume a simple x,y translation model is enough for alignment meaning we just need to shift the images up and down, left and right

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

# Here we go :p 
def ncc(img1, img2):
    img1 = img1.flatten()
    img2 = img2.flatten()
    vec1 = (img1 - np.mean(img1)) / (np.linalg.norm(img1 - np.mean(img1)))
    vec2 = (img2 - np.mean(img2)) / (np.linalg.norm(img2 - np.mean(img2)))
    return np.dot(vec1, vec2)

def align(top_img, base_img):
    height, width = base_img.shape
    crop_height, crop_width = int(height * 0.15), int(width * 0.15)
    crop_base_img = base_img[crop_height: height - crop_height, crop_width:width-crop_width]
    
    best_vector = np.array([0,0])
    best_metric = -np.inf
    align_img = top_img
    for i in np.arange(-15,16):
        for j in np.arange(-15,16):
            row_shifted = np.roll(top_img, shift=j, axis = 0) # axis 0 is rows, axis 1 is columns
            column_shifted = np.roll(row_shifted, shift = i, axis = 1)
            # To get better alignment lets crop out the black edges since they are messing up our metric
            crop_col_shifted = column_shifted[crop_height: height - crop_height, crop_width:width-crop_width]
            temp_metric = ncc(crop_col_shifted, crop_base_img)
            if temp_metric > best_metric:
                best_metric = temp_metric
                best_vector = np.array([j,i])
                align_img = column_shifted
    return align_img, best_vector

ag, ag_vec = align(g, b)
ar, ar_vec = align(r, b)

print(ag_vec)
print(ar_vec)

# create a color image
im_out = np.dstack([ar, ag, b])

# save the image
fname = f'/Users/kennethllontop/kennethllontop.github.io/proj_1/out_path/out_{pic}.jpg'
# I had an issue with the type of data I am passing in

# I need to keep values in range [0,1]
im_out = np.clip(im_out, 0, 1) 

#skio.imsave(fname, im_out)
skio.imsave(fname, sk.img_as_ubyte(im_out))

# display the image
skio.imshow(im_out)
skio.show()
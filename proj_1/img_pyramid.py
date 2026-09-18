# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

# Added this import so I can convolve
from scipy import signal

# name of the input file
pictures = ['emir.tif', 'harvesters.tif', 'icon.tif', 'ilemselga.tif', 'melons.tif', 'religous_painting.tif', 'self_portrait.tif', 'siren.tif', 'three_generations.tif', 'wharf.tif', "church.tif"]

# pictures = ['img1.jpg', 'img2.jpg', 'img3.jpg']

for pic in pictures:
    # imname = f'part4_img/{pic}'
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

    def align(top_img, base_img, starting_pt = (0, 0), search_range = 15):
        height, width = base_img.shape
        crop_height, crop_width = int(height * 0.15), int(width * 0.15)
        crop_base_img = base_img[crop_height: height - crop_height, crop_width:width-crop_width]
        
        best_vector = np.array(starting_pt)
        best_metric = -np.inf

        for i in np.arange(starting_pt[1] - search_range, starting_pt[1] + search_range + 1):
            for j in np.arange(starting_pt[0] - search_range, starting_pt[0] + search_range + 1):
                row_shifted = np.roll(top_img, shift=j, axis = 0) # axis 0 is rows, axis 1 is columns
                column_shifted = np.roll(row_shifted, shift = i, axis = 1)
                # To get better alignment lets crop out the black edges since they are messing up our metric
                crop_col_shifted = column_shifted[crop_height: height - crop_height, crop_width:width-crop_width]
                temp_metric = ncc(crop_col_shifted, crop_base_img)
                if temp_metric > best_metric:
                    best_metric = temp_metric
                    best_vector = np.array([j,i])
        return best_vector

    # For the image pyramid I am going to have to reduce the frequencies and downsample ie. blur and reduce image sixe by 1/2

    def gaussian_filter(img):
        kernel = np.array([[1, 2, 1,],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16
        blurry_img = signal.convolve2d(img, kernel, mode="same", boundary ="symm") # mode is same to keep the same dim; boundary is symm to not average edges to black
        return blurry_img

    # This code will be the whole blur and sub sampling process
    py_lvls = 5
    py_images = [[b, g, r]]

    new_b = b
    new_g = g
    new_r = r

    for i in np.arange(1, py_lvls): # already initialized it with the big image
        # I will need to subsample and blur b, g, r
        blurry_b = gaussian_filter(new_b)
        blurry_g = gaussian_filter(new_g)
        blurry_r =  gaussian_filter(new_r)

        down_b = blurry_b[0:-1:2,0:-1:2]
        down_g = blurry_g[0:-1:2,0:-1:2]
        down_r = blurry_r[0:-1:2,0:-1:2]

        bgr = [down_b, down_g, down_r]

        py_images.append(bgr)

        new_b = down_b
        new_g = down_g
        new_r = down_r

    # starting the recursion/iterations
    flag = 0
    for imagenes in reversed(py_images):
        if flag == 0:
            ag_vec = align(imagenes[1], imagenes[0])
            ar_vec = align(imagenes[2], imagenes[0])
            flag += 1
        else:
            ag_vec = align(imagenes[1], imagenes[0], 2 * ag_vec) # Account for the downsampling
            ar_vec = align(imagenes[2], imagenes[0], 2 * ar_vec)

    print(ag_vec)
    print(ar_vec)

    ag_row = np.roll(py_images[0][1], shift=ag_vec[0], axis = 0) # axis 0 is rows, axis 1 is columns
    ag = np.roll(ag_row, shift = ag_vec[1], axis = 1)

    ar_row = np.roll(py_images[0][2], shift=ar_vec[0], axis = 0) # axis 0 is rows, axis 1 is columns
    ar = np.roll(ar_row, shift = ar_vec[1], axis = 1)
    # create a color image
    im_out = np.dstack([ar, ag, py_images[0][0]])

    # save the image
    fname = f'/Users/kennethllontop/kennethllontop.github.io/proj_1/out_path/out_{pic}'
    # fname = f'/Users/kennethllontop/kennethllontop.github.io/proj_1/part4_out/out_{pic}.jpg'
    # I had an issue with the type of data I am passing in

    # I need to keep values in range [0,1]
    im_out = np.clip(im_out, 0, 1) 

    #skio.imsave(fname, im_out)
    skio.imsave(fname, sk.img_as_ubyte(im_out))

# display the image
# skio.imshow(im_out)
# skio.show()
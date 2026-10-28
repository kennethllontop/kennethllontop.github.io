# These are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import json # I need this to read the corresponding points
import matplotlib.pyplot as plt
import time
import os
import sys

np.set_printoptions(precision=3, suppress=True, linewidth=150)

os.makedirs('output_images', exist_ok=True)

def computeH(img1_pts, img2_pts, saving):
    '''
    im1_pts and im2_pts are n-by-2 matrices holding the (x,y) locations of 
    n point correspondences from the two images and H is the recovered 3x3 homography matrix.
    '''
    # I need to have an Ah = b and solve for h using least squares.
    A, b = [], []

    for (x,y), (x2, y2) in zip(img1_pts, img2_pts):
        A.append([x, y, 1, 0, 0, 0, -x*x2, -y*x2])
        A.append([0, 0, 0, x, y, 1, -x*y2, -y*y2])
        b.extend([x2, y2])
    h, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    h = np.append(h,1).reshape(3,3)
    if saving:
        with open(saving, 'w', encoding='utf-8') as file:
            file.write(f"A =\n{np.array(A)}\n\n")
            file.write(f"b =\n{np.array(b)}\n\n")
            file.write(f"h =\n{np.array(h)}\n\n")
    return h

def outline(im, H):
    height, width = im.shape[:2]
    edges = np.array([[0,0, 1],
                      [0, height - 1, 1],
                      [width - 1, 0, 1],
                      [width - 1, height - 1, 1]], dtype = float).T# I had to change this becasue of how H was expecting (x,y) => (column, row) or (width, height)
    new_edges = H @ edges
    new_edges = new_edges[:2]/new_edges[2] # Now we have x, y
    x_max, y_max = np.max(new_edges, axis = 1) # max across the rows
    x_min, y_min = np.min(new_edges, axis = 1) # min across the rows

    x_min, y_min = np.floor([x_min, y_min]).astype(int) # We do this to ensure that the whole image is actually being warped and not cutoff
    x_max, y_max = np.ceil([x_max, y_max]).astype(int)

    return x_min, x_max, y_min, y_max # These are the coordinates of the output image

def inverse_warping(im, H):
    x_min, x_max, y_min, y_max = outline(im, H)
    # I need to get all the output positions and multiply it by the inverse of H and then that would give use the source position
    # target_positions = []
    # for r in range(y_min, y_max + 1):
    #     for c in range(x_min, x_max + 1):
    #         target_positions.append([c, r, 1])
    # target_pos = np.array(target_positions).T # 3xN matrix where the rows are the x, y, and 1 values

    # I was running out of space on my computer so I had to run this
    xs, ys = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    target_pos = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)]) # 3xN matrix where the rows are the x, y, and 1 values
    source_pos = np.linalg.inv(H) @ target_pos # 3xN matrix where the rows are the wx, wy, and w values


    # We need to divide by w since each row represents the coordinate
    source_x = source_pos[0]/source_pos[2]
    source_y = source_pos[1]/source_pos[2]

    # We need to reshape the source to actual be a grid of the actual positions and the actual shape of the output image
    range_x, range_y = x_max - x_min + 1, y_max - y_min + 1
    source_x = source_x.reshape(range_y, range_x)
    source_y = source_y.reshape(range_y, range_x)
    return source_x, source_y, x_min, y_min

def warpImageNearestNeighbor(im, H):
    source_x, source_y, x_min, y_min = inverse_warping(im, H)
    x, y = np.round(source_x).astype(int), np.round(source_y).astype(int)
    # I have the source pixel locations and the the actual image. I need to make sure that the source points are actually on the image
    h, w = im.shape[:2]
    valid_mask = (x < w) & (x >= 0) & (y < h) & (y >= 0)
    return_img_y, return_img_x = source_x.shape 
    return_img = np.zeros((return_img_y, return_img_x, im.shape[2]))
    return_img[valid_mask] = im[y[valid_mask], x[valid_mask]]
    alpha = valid_mask.astype(float)
    return np.dstack([return_img, alpha]), (x_min, y_min) 

def warpImageBilinear(im, H):
    source_x, source_y, x_min, y_min = inverse_warping(im, H)
    # I have the source pixel locations and the the actual image. I need to make sure that the source points are actually on the image
    h, w = im.shape[:2]
    valid_mask = (source_x <= w-1) & (source_x >= 0) & (source_y <= h-1) & (source_y >= 0) # Different from NN because the source locations there have already been rounded to whole numbers; we are doing this before the round and looking for all 4 picels to exist
    x, y = source_x[valid_mask], source_y[valid_mask]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int) # We do be looking for the pixels, upper left
    x1, y1 = np.minimum(x0 + 1, w-1), np.minimum(y0 + 1, h-1) # Looking again, lower right
    a = (x-x0)[:, None] # getting the weights and turning it into a column
    b = (y-y0)[:, None]
    val = ((1-a) * (1-b) * im[y0, x0] + a * (1-b) * im[y0, x1] + (1-a) * b * im[y1, x0] + a * b * im[y1, x1]) # holds los colores
    return_img_y, return_img_x = source_x.shape 
    return_img = np.zeros((return_img_y, return_img_x, im.shape[2]))
    return_img[valid_mask] = val
    alpha = valid_mask.astype(float)
    return np.dstack([return_img, alpha]), (x_min, y_min) 

def rectification(pic, json_name, W, Ht, rotate=True):
    name = pic.split('.')[0]
    imname = f'images/{pic}'
    
    # read in the image
    im = skio.imread(imname)

    im = sk.img_as_float(im)

    im = im[..., :3]

    #Display the image correctly
    if rotate:
        im = np.rot90(im, k=-1)

    with open(json_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    pts = np.array(data['im_points'], dtype=float)
    target = np.array([[0, 0], [W, 0], [W, Ht], [0, Ht]], dtype=float)

    H = computeH(pts, target, saving=f'output_images/{name}_info.txt')

    start = time.perf_counter()
    rect_nn, _ = warpImageNearestNeighbor(im, H)
    t_nn = time.perf_counter() - start

    start = time.perf_counter()
    rect_bil, _ = warpImageBilinear(im, H)
    t_bil = time.perf_counter() - start

    with open(f'output_images/{name}_info.txt', 'a', encoding='utf-8') as file:
        file.write(f"Nearest neighbor: {t_nn:.2f}s   Bilinear: {t_bil:.2f}s\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(im);                axes[0].set_title('Original')
    axes[1].imshow(rect_nn[..., :3]);  axes[1].set_title('Nearest neighbor')
    axes[2].imshow(rect_bil[..., :3]); axes[2].set_title('Bilinear')
    for ax in axes:
        ax.axis('off')
    fig.savefig(f'output_images/{name}_rectification.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
 
    plt.imsave(f'output_images/{name}_nn.png', np.clip(rect_nn[..., :3], 0, 1))
    plt.imsave(f'output_images/{name}_bil.png', np.clip(rect_bil[..., :3], 0, 1))

    return rect_nn, rect_bil

def weight_map(im):
    h, w = im.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w] # ys holds the row index, and xs holds the column index
    dist = np.minimum.reduce([xs, w - 1 - xs, ys, h - 1 - ys]).astype(float)
    return (dist + 1) / (dist.max() + 1)


rect_nn1, rect_bil1 = rectification('comp_screen.png', 'computer_screen.json', 1600, 1000)
rect_nn2, rect_bil2 = rectification('portrait_mom.png', 'painting.json', 1100, 1400)

# name of the input file
pictures1 = ['building_8_1.png', 'building_9_1.png', 'stairs_1.png']
pictures2 = ['building_8_2.png', 'building_9_2.png', 'stairs_2.png']
rotations = [False, False, False] # This no longer became necessary once I shrunk the images

for pic1, pic2, rotate in zip(pictures1, pictures2, rotations):
    
    imname1 = f'images/{pic1}'
    imname2 = f'images/{pic2}'
    name1, name2 = pic1.split('.')[0], pic2.split('.')[0]
    
    # read in the image
    im1 = skio.imread(imname1)
    im2 = skio.imread(imname2)

    # convert to double (might want to do this later on to save memory)    
    im1 = sk.img_as_float(im1)
    im2 = sk.img_as_float(im2)

    im1, im2 = im1[..., :3], im2[..., :3]

    #Display the image correctly
    if rotate:
        im1 = np.rot90(im1, k=-1)
        im2 = np.rot90(im2, k=-1)

    with open(f'{name1}_{name2}.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    pts1, pts2 = np.array(data['im1Points'], dtype=float), np.array(data['im2Points'], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, im, pts, title in zip(axes, [im1, im2], [pts1, pts2], ['room1', 'room2']):
        ax.imshow(im)
        ax.scatter(pts[:, 0], pts[:, 1], c='red', s=30)
        for i, (x, y) in enumerate(pts):
            ax.text(x + 10, y - 10, str(i), color='yellow', fontsize=12)
        ax.set_title(title)
        ax.axis('off')
    fig.savefig(f'output_images/{name1}_{name2}_points.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
 
    h = computeH(pts1, pts2, saving=f'output_images/{name1}_{name2}_info.txt')

    im1_wt = np.dstack([im1, weight_map(im1)])
    im2_wt = np.dstack([im2, weight_map(im2)])
    
    # I need to figure out the size of the canvas; need to take into account the smallest and largest of unwarped and warped image
    start = time.perf_counter()
    warped_output, (x_min, y_min) = warpImageBilinear(im1_wt, h) # x_min and y_min are the smalled coordinates in the output
    t_warp = time.perf_counter() - start
    warped_h, warped_w = warped_output.shape[:2]
    h2, w2 = im2.shape[:2]

    min_x, min_y = min(x_min, 0), min(y_min, 0) # Find out smallest coordinate canvas needs to cover, unwarped image smallest coordiante is 0,0
    max_x = max(x_min + warped_w - 1, w2 - 1)
    max_y = max(y_min + warped_h - 1, h2 - 1)

    range_y, range_x = max_y - min_y + 1, max_x - min_x + 1
    cav1 = np.zeros((range_y, range_x, warped_output.shape[2]))
    row1, col1 = y_min - min_y, x_min - min_x
    cav1[row1:row1+ warped_h, col1: col1 + warped_w] = warped_output

    cav2 = np.zeros((range_y, range_x, im2_wt.shape[2]))
    row2, col2 = - min_y, - min_x
    cav2[row2:row2 + h2, col2: col2 + w2] = im2_wt

    w1 = cav1[..., 3] * cav1[..., 4]
    w2 = cav2[..., 3]
    total = w1 + w2
    total[total == 0] = 1
    return_img = (cav1[..., :3] * w1[..., None] + cav2[..., :3] * w2[..., None]) / total[..., None] # This is hte weight average

    # THis is no blending, im1 is warped and then img2 is pasted on top
    overwrite = cav1[..., :3].copy()
    im2_present = cav2[..., 3] > 0
    overwrite[im2_present] = cav2[..., :3][im2_present]

    with open(f'output_images/{name1}_{name2}_info.txt', 'a', encoding='utf-8') as file:
        file.write(f"Warp time: {t_warp:.2f}s\n")
        file.write(f"Offset (x_min, y_min): ({x_min}, {y_min})\n")
        file.write(f"Canvas size (height, width): ({range_y}, {range_x})\n")
 
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(overwrite);  axes[0].set_title('Overwrite (no blending)')
    axes[1].imshow(return_img); axes[1].set_title('Weighted averaging')
    for ax in axes:
        ax.axis('off')
    fig.savefig(f'output_images/{name1}_{name2}_blending.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
 
    plt.imsave(f'output_images/{name1}_{name2}_mosaic.png', np.clip(return_img, 0, 1))


    


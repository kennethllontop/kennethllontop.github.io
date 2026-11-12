
import numpy as np
from skimage.feature import corner_harris, peak_local_max
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
import json # I need this to read the corresponding points
from skimage.color import rgb2gray
import os

os.makedirs('output_pictures', exist_ok=True)

def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

def anms(im, num_interest=500):
    h, coords = get_harris_corners(im)
    ys, xs = coords[0], coords[1]
    strength_harris = h[ys, xs]

    coordinates = coords.T
    distances = dist2(coordinates, coordinates)

    c_robust = 0.9

    r_i = []

    for i in range(len(distances)):
        list_j = []
        for j in range(len(distances)):
            if strength_harris[i] < c_robust * strength_harris[j]:
                list_j += [j]
        valid = []
        for j in list_j:
            valid += [distances[:, i][j]]
        if len(valid) > 0:
            r_i += [min(valid)]
        else:
            r_i += [np.inf]

    r_i = np.array(r_i)
    keep = np.argsort(-r_i)[:num_interest]
    return coords[:, keep]

def feature_descriptor(gray_im, name, coords):
    # I need to build a gaussian kernel to blur the image and then keep every fifth pixel 
    # Create Gaussian Kernel
    k = 13
    g_kernel = cv2.getGaussianKernel(ksize=k, sigma=2)
    g_kernel_2d = g_kernel @ g_kernel.T

    # Getting the blur and high frequencies
    blur_img = convolve2d(gray_im, g_kernel_2d, mode="same", boundary ="symm")

    # Blurry Image
    cv2.imwrite(f'output_pictures/blur_{name}.png', (blur_img * 255).astype(np.uint8))

    # Now that I have the blurry image I need to sample surrounding the interest points

    descriptors = []

    for y, x  in coords.T:
        window = blur_img[y-20:y+20, x-20:x+20]

        # Now get every fifth column and row
        patch = window[::5,::5]

        # Need to make the window have mean 0 and std 1
        mean_patch = np.mean(patch)
        std_patch = np.std(patch)

        normal_patch = (patch - mean_patch) / std_patch

        final_patch = normal_patch.flatten()

        descriptors += [final_patch]

    return np.array(descriptors)

def feature_matching(descriptor1, descriptor2):
    # A good threshold based on the table seems to be 0.2
    threshold_limit = 0.2

    # How can I tell if descriptors are similar then the distance between them must be small.
    #Descriptor1 holds all the patches for image 1 and descriptor 2 holds all the patches for image 2
    distance = dist2(descriptor1, descriptor2)

    order_distances = np.argsort(distance, axis=1)
    
    nn1 = order_distances[:,0] # index of besst match in image 2
    nn2 = order_distances[:,1]

    rows = np.arange(len(distance))

    e1 = distance[rows, nn1]
    e2 = distance[rows, nn2]
    
    ratio = e1/e2

    mask = ratio < threshold_limit

    return np.stack([rows[mask], nn1[mask]], axis=1) # We are returning the patches in image 1 with who they are closest to in image 2; row is 1 ex - the patch in image 1 is column 1 and the assocated pair in img 2 is column2

def ransac (feat_match, coords1, coords2, iterations=2000):
    pts1_all = coords1[:, feat_match[:, 0]].T[:, ::-1]
    pts2_all = coords2[:, feat_match[:, 1]].T[:, ::-1]

    n = len(pts1_all)
    pts1_h = np.hstack([pts1_all, np.ones((n, 1))]) 

    best_inliers = np.zeros(len(feat_match), dtype=bool)

    for i in range(iterations):
        idx = np.random.choice(len(feat_match), size=4, replace=False)
        samples = feat_match[idx]

        pts_h_1 = []
        pts_h_2 = []

        for sam_img1, sam_img2 in samples:
            y1, x1 = coords1[:, sam_img1]
            y2, x2 = coords2[:, sam_img2]

            pts_h_1.append([x1, y1])
            pts_h_2.append([x2, y2])

        H = computeH(np.array(pts_h_1), np.array(pts_h_2), saving=None)

        projected = (H @ pts1_h.T).T 
        projected = projected[:, :2] / projected[:, 2:3] 
        dist = np.sqrt(np.sum((projected - pts2_all) ** 2, axis=1))

        inliers = dist < 3
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    H = computeH(pts1_all[best_inliers], pts2_all[best_inliers], saving=None)
    return H, best_inliers



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

def weight_map(im):
    h, w = im.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w] # ys holds the row index, and xs holds the column index
    dist = np.minimum.reduce([xs, w - 1 - xs, ys, h - 1 - ys]).astype(float)
    return (dist + 1) / (dist.max() + 1)

def make_mosiac(im1, im2, h):
    im1_wt = np.dstack([im1, weight_map(im1)])
    im2_wt = np.dstack([im2, weight_map(im2)])
    
    # I need to figure out the size of the canvas; need to take into account the smallest and largest of unwarped and warped image
    warped_output, (x_min, y_min) = warpImageBilinear(im1_wt, h) # x_min and y_min are the smalled coordinates in the output
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
    return (cav1[..., :3] * w1[..., None] + cav2[..., :3] * w2[..., None]) / total[..., None] # This is hte weight average

# Getting my cute little pictures
def plot_corners(gray, coords, title, path):
    plt.figure(figsize=(8, 6))
    plt.imshow(gray, cmap='gray')
    plt.scatter(coords[1], coords[0], s=4, c='red')   # xs first, since coords is (y, x)
    plt.title(f"{title} ({coords.shape[1]} corners)")
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_descriptors(gray, coords, descriptors, path, num=6):
    picks = np.random.choice(coords.shape[1], num, replace=False)
    fig, axes = plt.subplots(2, num, figsize=(2 * num, 4))
    for col, k in enumerate(picks):
        y, x = coords[:, k]
        axes[0, col].imshow(gray[y-20:y+20, x-20:x+20], cmap='gray')
        axes[1, col].imshow(descriptors[k].reshape(8, 8), cmap='gray')
        axes[0, col].set_title(f"({x}, {y})", fontsize=8)
        axes[0, col].axis('off')
        axes[1, col].axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_matches(gray1, gray2, coords1, coords2, matches, title, path):
    h = max(gray1.shape[0], gray2.shape[0])
    pad1 = np.pad(gray1, ((0, h - gray1.shape[0]), (0, 0)))
    pad2 = np.pad(gray2, ((0, h - gray2.shape[0]), (0, 0)))
    combined = np.hstack([pad1, pad2])
    offset = gray1.shape[1]                      # shift image-2 points right

    plt.figure(figsize=(14, 6))
    plt.imshow(combined, cmap='gray')
    for i, j in matches:
        y1, x1 = coords1[:, i]
        y2, x2 = coords2[:, j]
        plt.plot([x1, x2 + offset], [y1, y2], linewidth=0.7)
        plt.scatter([x1, x2 + offset], [y1, y2], s=6)
    plt.title(f"{title} ({len(matches)} matches)")
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def compare_mosaics(manual, auto, path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, img, title in zip(axes, [manual, auto], ["Manual", "Automatic"]):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)


pictures1 = ['building_8_1.png', 'building_9_1.png', 'indoor_house_1.png']
pictures2 = ['building_8_2.png', 'building_9_2.png', 'indoor_house_2.png']

for pic1, pic2 in zip(pictures1, pictures2):
    name1, name2 = pic1.split('.')[0], pic2.split('.')[0]
    im1 = sk.img_as_float(skio.imread(f'images/{pic1}'))[..., :3]
    im2 = sk.img_as_float(skio.imread(f'images/{pic2}'))[..., :3]

    # Manual H from Part A's clicked points
    with open(f'{name1}_{name2}.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    H_manual = computeH(np.array(data['im1Points'], dtype=float),
                        np.array(data['im2Points'], dtype=float), saving=None)

    # Automatic H from the Part B pipeline
    gray1, gray2 = rgb2gray(im1), rgb2gray(im2)
    coords1, coords2 = anms(gray1), anms(gray2)
    desc1 = feature_descriptor(gray1, name1, coords1)
    desc2 = feature_descriptor(gray2, name2, coords2)
    matches = feature_matching(desc1, desc2)
    H_auto, inliers = ransac(matches, coords1, coords2)

    print(f"{name1}: {len(matches)} matches, {inliers.sum()} inliers")
    print("H_manual:\n", H_manual, "\nH_auto:\n", H_auto)

    # B.1 to B.4 figures
    _, all_corners = get_harris_corners(gray1)
    plot_corners(gray1, all_corners, "Harris corners", f"output_pictures/{name1}_harris.png")
    plot_corners(gray1, coords1, "After ANMS", f"output_pictures/{name1}_anms.png")
    plot_descriptors(gray1, coords1, desc1, f"output_pictures/{name1}_descriptors.png")
    plot_matches(gray1, gray2, coords1, coords2, matches, "All matches", f"output_pictures/{name1}_matches.png")
    plot_matches(gray1, gray2, coords1, coords2, matches[inliers], "RANSAC inliers", f"output_pictures/{name1}_inliers.png")

    # Manual vs automatic mosaics
    mosaic_manual = make_mosiac(im1, im2, H_manual)
    mosaic_auto = make_mosiac(im1, im2, H_auto)
    compare_mosaics(mosaic_manual, mosaic_auto, f"output_pictures/{name1}_{name2}_comparison.png")
    plt.imsave(f"output_pictures/{name1}_{name2}_auto_mosaic.png", np.clip(mosaic_auto, 0, 1))
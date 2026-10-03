import numpy as np
from scipy.signal import convolve2d # Built-in function
import cv2 # Using opencv for the grayscale images

# Part 1.1 Implement convolutions from scratch
# Same Padding means you add enough zeros that the output has the same size as the input, (k-1)/2 on each side
# Full Padding means you add k-1 on each side

def four_for_loops(input_img, kernal, padding):
    k_height, k_width = kernal.shape
    output = []
    flipped = np.flip(kernal)

    if padding == 'same':
        pad = ((k_height // 2, (k_height - 1) // 2),(k_width // 2, (k_width - 1) // 2))
        img = np.pad(np.asarray(input_img, dtype=np.float64), pad_width = pad, mode='constant', constant_values=0)
    elif padding == 'full':
        pad = ((k_height - 1, k_height - 1),(k_width - 1, k_width - 1))
        img = np.pad(np.asarray(input_img, dtype=np.float64), pad_width = pad, mode='constant', constant_values=0)
    else:
        img = np.asarray(input_img, dtype=np.float64)  # Valid Padding

    for height_idx_img in range(0, len(img)-k_height+1):
        collection_sums = []
        for width_idx_img in range(0, len(img[0])-k_width+1):
            temp_sum = 0
            for height_idx_filter in range(0, k_height):
                for width_idx_filter in range(0, k_width):
                    answer = flipped[height_idx_filter, width_idx_filter] * img[height_idx_img + height_idx_filter, width_idx_img + width_idx_filter]
                    temp_sum += answer
            collection_sums.append(temp_sum)
        output.append(collection_sums)
    return np.array(output)

def two_for_loops(input_img, kernal, padding):
    k_height, k_width = kernal.shape
    output = []
    flipped = np.flip(kernal)

    if padding == 'same':
        pad = ((k_height // 2, (k_height - 1) // 2),(k_width // 2, (k_width - 1) // 2))
        img = np.pad(np.asarray(input_img, dtype=np.float64), pad_width = pad, mode='constant', constant_values=0)
    elif padding == 'full':
        pad = ((k_height - 1, k_height - 1),(k_width - 1, k_width - 1))
        img = np.pad(np.asarray(input_img, dtype=np.float64), pad_width = pad, mode='constant', constant_values=0)
    else:
        img = np.asarray(input_img, dtype=np.float64)  # Valid Padding

    for height_idx_img in range(0, len(img)-k_height+1):
        collection_sums = []
        for width_idx_img in range(0, len(img[0])-k_width+1):
            # Element wise multiplication and then sum
            patch = img[height_idx_img:height_idx_img+k_height, width_idx_img:width_idx_img+k_width]
            temp_sum = np.sum(patch*flipped)
            collection_sums.append(temp_sum)
        output.append(collection_sums)
    return np.array(output)

# Comparing it with the built-in convolution function
def testing():
    img = np.random.rand(6, 6)
    for shape in [(2, 2), (3, 3), (4, 4), (5, 5), (1, 2), (2, 1), (2, 3), (3, 1)]:
        kernel = np.random.rand(*shape)
        for mode in ['same', 'full', 'valid']:
            expected = convolve2d(img, kernel, mode=mode)
            assert np.allclose(four_for_loops(img, kernel, mode), expected)
            assert np.allclose(two_for_loops(img, kernel, mode), expected)
    print("All tests passed")

testing()

# Process img as grayscale
pic = 'kenneth.jpg'
gray_img = cv2.imread(f'pictures/{pic}', 0)

# cv2.imshow("image", gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Write out 9x9 box filter
box_9_x_9 = np.ones((9,9)) / 81

# Write out difference operators
diff_x = np.array([[1, 0, -1]])
diff_y = np.array([[1], [0], [-1]])

# convolve the picture with the box filter and difference operators

cv2.imwrite('output_pictures/box_9_x_9_kenneth.png', two_for_loops(gray_img, box_9_x_9, 'same').astype(np.uint8))

partial_dx = (two_for_loops(gray_img, diff_x, 'same') + 255) /2
partial_dy = (two_for_loops(gray_img, diff_y, 'same') + 255) /2

cv2.imwrite('output_pictures/diff_x_kenneth.png', partial_dx.astype(np.uint8))
cv2.imwrite('output_pictures/diff_y_kenneth.png', partial_dy.astype(np.uint8))
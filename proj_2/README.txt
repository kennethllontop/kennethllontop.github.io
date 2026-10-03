CS180 Project 2: Fun with Filters and Frequencies
Kenneth Llontop


HOW TO RUN

Run everything from the proj_2 folder:

    python main.py

Run only some parts by listing them:

    python main.py 1.1 2.4

The parts are 1.1, 1.2, 1.3, 2.1, 2.2, 2.3 and 2.4. With no parts listed, main.py runs
them in the order 1.1, 1.2, 1.3, 2.1, 2.3, 2.4, 2.2.


PACKAGES

Python 3.8 with numpy, scipy, opencv-python, scikit-image and matplotlib.

    pip install numpy scipy opencv-python scikit-image matplotlib


WHICH FILE IS WHICH PART

fun_filters_1_1.py   Part 1.1  Convolution from scratch with four and two for loops,
                               compared with scipy.signal.convolve2d, then a 9x9 box filter
                               and the finite difference operators on a picture of me
fun_filters_1_2.py   Part 1.2  Finite difference operator, gradient magnitude and
                               binarized edges of the cameraman
fun_filters_1_3.py   Part 1.3  Gaussian blur then finite differences, the DoG filters, and a
                               check that both give the same result
fun_filters_2_1.py   Part 2.1  Unsharp mask filter, blurred and high frequency images,
                               alpha 1, 2, 4 and 8, and the blur then sharpen evaluation
cs180_proj2_hybrid_starter_code/hybrid_image_starter.py
                     Part 2.2  Hybrid images. align_image_code.py is the provided alignment
                               code and is only imported
fun_filters_2_3.py   Part 2.3  Gaussian and Laplacian stacks of the apple and orange
fun_filters_2_4.py   Part 2.4  Multiresolution blending and the masked Laplacian stack figures


INPUTS AND OUTPUTS

Input pictures are in proj_2/pictures and in proj_2/cs180_proj2_hybrid_starter_code.

Results are saved to proj_2/output_pictures, proj_2/output_pictures/fig_2_1 and
proj_2/cs180_proj2_hybrid_starter_code/output_pictures. main.py makes these folders if they
are missing.


THINGS TO KNOW WHILE IT RUNS

- Part 2.2 asks you to click alignment points for each of the 3 hybrids. For the Toothless and Stitch results
  I clicked on the white part of Stitches eyes and the I clicked Toothless eyes. However for Toothless Left eye
  click horizontally to left most spot and the right eye click on North East part of the eye. Soup and Empty
  I just clicked on the North and South edges of the plate. For Derek and Nutmeg I clicked in the middle of their
  pupils. 
- The hybrid code shows its results with plt.show(). Close each window to continue.
- Part 1.1 prints "All tests passed" after comparing my convolutions with
  scipy.signal.convolve2d.
- Part 1.3 prints the difference between blur then difference and the DoG filters,
  ignoring the outer 2 pixels.


SWITCHING BLENDS AND HYBRID PAIRS

main.py passes the settings to the scripts, so no code needs to change. All 3 blends and
all 3 hybrid pairs run by default.

Each blend in the blends list in main.py has:
    pictures      the two pictures to blend, the first one is kept where the mask is 1
    mask_type     vertical, horizontal or person, person uses pictures/person_mask.png
    out_name      name of the saved blend and of its Laplacian stack figure

Each hybrid in the hybrids list in main.py has:
    high_file     the image you see up close
    low_file      the image you see far away
    high_name     name used for the saved aligned and high pass images of high_file
    low_name      name used for the saved aligned and low pass images of low_file
    hybrid_name   name of the saved hybrid and of its grayscale version
    sigma1        cutoff for the high pass image
    sigma2        cutoff for the low pass image
    make_fft      True to also save the Fourier transform figure, used for Toothless and Stitch

To add a blend or a pair, add another entry to its list. Running a script directly without
main.py uses its default settings, which are the oraple for 2.4 and Derek and Nutmeg for 2.2.

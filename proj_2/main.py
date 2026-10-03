import os
import sys
import runpy

# Run from the proj_2 folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# cv2.imwrite does not make folders
os.makedirs('output_pictures/fig_2_1', exist_ok=True)
os.makedirs('cs180_proj2_hybrid_starter_code/output_pictures', exist_ok=True)

# Settings for each blend in part 2.4
blends = [
    {'pictures': ['apple', 'orange'], 'mask_type': 'vertical', 'out_name': 'oraple'},
    {'pictures': ['night', 'ocean'], 'mask_type': 'horizontal', 'out_name': 'night_ocean'},
    {'pictures': ['blue_shell', 'water'], 'mask_type': 'person', 'out_name': 'cold_water'},
]

# Settings for each hybrid in part 2.2, high is the image you see up close and low is the image you see far away
hybrids = [
    {'high_file': './nutmeg.jpg', 'low_file': './DerekPicture.jpg', 'high_name': 'nutmeg', 'low_name': 'derek',
     'hybrid_name': 'cat_derek_hybrid', 'sigma1': 10, 'sigma2': 15, 'make_fft': False},
    {'high_file': './soup.jpg', 'low_file': './empty.jpg', 'high_name': 'soup', 'low_name': 'empty',
     'hybrid_name': 'soup_hybrid', 'sigma1': 2, 'sigma2': 6, 'make_fft': False},
    {'high_file': './stitch.jpg', 'low_file': './toothless.jpg', 'high_name': 'stitch', 'low_name': 'toothless',
     'hybrid_name': 'toothless_hybrid', 'sigma1': 2, 'sigma2': 6, 'make_fft': True},
]

scripts = {'1.1': 'fun_filters_1_1.py', '1.2': 'fun_filters_1_2.py', '1.3': 'fun_filters_1_3.py',
           '2.1': 'fun_filters_2_1.py', '2.3': 'fun_filters_2_3.py'}
all_parts = ['1.1', '1.2', '1.3', '2.1', '2.3', '2.4', '2.2']

# Parts to run, all of them if none are given
parts = sys.argv[1:] or all_parts
for part in parts:
    if part not in all_parts:
        sys.exit(f'Unknown part {part}, choose from 1.1 1.2 1.3 2.1 2.2 2.3 2.4')

for part in parts:
    print(f'Running part {part}')
    if part in scripts:
        runpy.run_path(scripts[part], run_name='__main__')
    elif part == '2.4':
        for blend in blends:
            print(f'  Blending {blend["out_name"]}')
            runpy.run_path('fun_filters_2_4.py', init_globals=blend, run_name='__main__')
    elif part == '2.2':
        # The hybrid code loads its images and the align code from its own folder
        os.chdir('cs180_proj2_hybrid_starter_code')
        sys.path.insert(0, os.getcwd())
        for hybrid in hybrids:
            print(f'  Making {hybrid["hybrid_name"]}')
            runpy.run_path('hybrid_image_starter.py', init_globals=hybrid, run_name='__main__')
        os.chdir('..')

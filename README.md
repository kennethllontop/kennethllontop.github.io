## Hi there 👋

## Project 1

Please clone the repo

```bash
git clone https://github.com/kennethllontop/kennethllontop.github.io.git
```

## Single-scale Alignment

```bash
cd kennethllontop.github.io/proj1

```
Open starter_code.py

# Change the pictures variable to select the images to align from the dataset
pictures = ['cathedral.jpg', 'monastery.jpg', 'tobolsk.jpg']

# Change imname to be where the images are
imname = f'CS180_fa2026_proj1_data/{pic}'

# Change fname to select the out put destination
fname = f'/Users/kennethllontop/kennethllontop.github.io/proj_1/out_path/out_{pic}.jpg'

# When ready run:
python starter_code.py

## Multi-scale Alignment

Open img_pyramid.py

# Change the pictures variable to select the images to align from the dataset
pictures = ['emir.tif', 'harvesters.tif', 'icon.tif', 'ilemselga.tif', 'melons.tif', 'religous_painting.tif', 'self_portrait.tif', 'siren.tif', 'three_generations.tif', 'wharf.tif', "church.tif"]

# Change imname to be where the images are
imname = f'CS180_fa2026_proj1_data/{pic}'

# Change fname to select the out put destination
fname = f'/Users/kennethllontop/kennethllontop.github.io/proj_1/out_path/out_{pic}'

# When ready run:
python img_pyramid.py

## Additional Results

Open img_pyramid.py

# Change the pictures variable to select the images to align from the dataset
pictures = ['img1.jpg', 'img2.jpg', 'img3.jpg']

# Change imname to be where the images are
imname = f'part4_img/{pic}'

# Change fname to select the out put destination
fname = f'/Users/kennethllontop/kennethllontop.github.io/proj_1/part4_out/out_{pic}'

# When ready run:
python img_pyramid.py

## Cropping

To study the effect of no cropping go to the align function

Uncomment:
temp_metric = ncc(column_shifted, base_img)

Comment:
crop_col_shifted = column_shifted[crop_height: height - crop_height, crop_width:width-crop_width]
temp_metric = ncc(crop_col_shifted, crop_base_img)

This is only found in the starter_code.py


<!--
**kennethllontop/kennethllontop** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

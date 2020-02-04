from Generic import filedialogs, images

direc = filedialogs.open_directory()
files = filedialogs.get_files_directory(direc + '/*.png')

final_image = []

for file in files:
    im = images.load(file)
    im = images.bgr_2_grayscale(im)
    final_image.append(im)

# %%
mean_image = images.mean(final_image)
images.save(mean_image, direc + '/mean_image.png')

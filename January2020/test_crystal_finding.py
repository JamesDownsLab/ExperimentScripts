from Generic import images

im = images.load("/media/data/Data/Logs/2020_1_15_15_29/0_mean.png", 0)

# %%
images.display(im)

# %%
# images.ThresholdGui(im)
# %%
threshold = images.threshold(im, 200)
images.display(threshold)

# %%
dilate = images.dilate(threshold, (5, 5))
# dilate = threshold
images.display(dilate)

# %%
opening = images.opening(dilate, (13, 13))
opening = images.opening(dilate, (21, 21))
images.display(opening)

# %%
center = images.center_of_mass(threshold)
im0 = images.draw_circle(im, center[0], center[1], 5)
im0 = images.draw_circle(im0, im0.shape[1] // 2, im0.shape[0] // 2, 5,
                         color=images.BLUE)
images.display(im0)

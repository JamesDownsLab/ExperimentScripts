from math import degrees, atan

import cv2
import numpy as np
from scipy import spatial
from scipy.ndimage import interpolation
from tqdm import tqdm

from Generic import images, filedialogs
from ParticleTracking import dataframes, statistics


def run_analysis(files, savename):
    N = len(files)
    ims = [images.load(f, 0) for f in files]

    # Find circles in all images
    circles = [images.find_circles(im, 27, 200, 7, 16, 16) for im in ims]

    # Save this information to a dataframe
    data = dataframes.DataStore(savename, load=False)
    for f, info in enumerate(circles):
        data.add_tracking_data(f, info, ['x', 'y', 'r'])

    # Calculate the order parameter for this data
    calc = statistics.PropertyCalculator(data)
    calc.order()

    # Find the angle of the images from the first image
    angle, xmin, xmax, h = get_im_angle(ims[0])
    L = xmax - xmin  # pixels, equal to 195mm if the edge of the balls is selected

    # Get the dh's for each frame
    dh = [get_dh(data.df.loc[f], ims[f], angle, xmin, xmax, h) for f in
          tqdm(range(5))]
    return dh, L


def get_dh(data, im, angle, xmin, xmax, h):
    field, x, y = coarse_order_field(data)
    phi_d, phi_o = get_phis(field)
    criteria = (phi_d + phi_o) / 1.6
    binary = get_binary_image(field, criteria)

    # Find the angle of the image
    # angle = get_im_angle(im)

    # Rotate the image then dilate it
    binary = interpolation.rotate(binary, angle)
    binary = np.uint8(binary)
    binary = images.dilate(binary, (5, 5))

    # Filter the image
    binary = filter_by_contours(binary)

    # Select border
    # xmin, xmax, h = get_border(interpolation.rotate(im, angle))

    # Find closest points to boundary on binary image
    x, y = get_boundary(binary, xmin, xmax, h)
    # plt.figure()
    # plt.imshow(interpolation.rotate(im, angle))
    # plt.plot(x, y)
    # plt.show()

    # Find dh
    dh = np.array(y) - h
    return dh.squeeze()


def get_boundary(binary, xmin, xmax, h):
    x_out = []
    y_out = []
    # y = np.arange(0, binary.shape[0])
    for x in range(xmin, xmax):
        row = binary[:, x]
        y_values = np.argwhere(row == 1)
        if len(y_values) > 0:
            dist = (y_values - h) ** 2
            smallest = np.argmin(dist)
            x_out.append(x)
            y_out.append(y_values[smallest])

    return x_out, y_out


def filter_by_contours(im):
    contours = images.find_contours(im)
    contours = images.sort_contours(contours)
    largest_contour = contours[-1]

    mask = np.zeros_like(im)
    cv2.fillConvexPoly(mask, largest_contour, (255, 255, 255))
    return images.mask_img(im, mask)


def get_im_angle(im):
    ls = LineSelector(im)
    p1, p2 = ls.points
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = p2[1] - m * p2[0]
    angle = degrees(atan(m))
    im = interpolation.rotate(im, angle)
    ls2 = LineSelector(im)
    xmin, xmax, h = ls2.points[0][0], ls2.points[1][0], ls2.points[0][1]
    return angle, xmin, xmax, h


def get_binary_image(field, criteria):
    error = 0.02 * criteria
    binary = (field > criteria - error) * (field < criteria + error)
    return binary


def get_phis(field, threshold=1e-5):
    s = field.shape
    d_values = field[:s[0] // 4, :]
    o_values = field[3 * s[0] // 4:, :]
    phi_d = np.mean(d_values[d_values > threshold])
    phi_o = np.mean(o_values[o_values > threshold])
    return phi_d, phi_o


def coarse_order_field(df):
    order = df.order.values
    x = np.arange(0, max(df.x), 1)
    y = np.arange(0, max(df.y), 1)
    x, y = np.meshgrid(x, y)
    r = np.dstack((x, y))

    # points_for_tree = df.loc[df.order > 0.8, ['x', 'y']].values
    points_for_tree = df[['x', 'y']].values
    tree = spatial.cKDTree(points_for_tree)
    tree_dists, tree_indxs = tree.query(r, 20, n_jobs=-1)
    cgw = np.mean(tree_dists[:, :, 0])

    exp_term = np.exp(-tree_dists ** 2 / (2 * cgw ** 2)) / (
            2 * np.pi * cgw ** 2)
    return np.sum(exp_term * order[tree_indxs], axis=2), x, y


class LineSelector:
    def __init__(self, im):
        cv2.namedWindow('line', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('line', 960, 540)
        cv2.setMouseCallback('line', self.record)
        self.points = []
        while True:
            cv2.imshow('line', im)
            key = cv2.waitKey(1) & 0xFF
            if len(self.points) == 2:
                break
        cv2.destroyAllWindows()

    def record(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])


if __name__ == "__main__":
    direc = filedialogs.open_directory()
    files = filedialogs.get_files_directory(direc + '/*.png')
    savename = direc + '/data.hdf5'
    dh, L = run_analysis(files, savename)

# %%
import matplotlib.pyplot as plt

dh, L = dh
dh_all = np.concatenate(dh)
plt.plot(dh_all)

# %%
sp = [np.fft.fft(d) for d in dh if len(d) == 1657]
freq = np.fft.fftfreq(len(dh[0]))

# %%
y = np.stack(sp)
y = np.mean(y, axis=0)

# %%
plt.loglog(freq, y * L)
plt.xlabel('$ k [pix^{-1}]$')
plt.ylabel(r'$ \langle | \delta h_k|^2\rangle L [pix^{-3}]$')

# %%
# plt.plot(np.log(freq), np.log(L*y**2))

N = len(dh[0])
freq_log = np.log(freq[1:N // 2])
y_log = np.log(L * np.abs(y[1:N // 2]) ** 2)

plt.plot(freq_log, y_log)

p, cov = np.polyfit(freq_log[10:], y_log[10:], 1, cov=True)
p1 = np.poly1d(p)

plt.plot(freq_log[10:], p1(freq_log[10:]), '-')

print("Gradient of Line = {:.2f} +/- {:.2f}".format(p[0], cov[0][0] ** 0.5))
print("Intercept at {:.2f}".format(p1[0]))

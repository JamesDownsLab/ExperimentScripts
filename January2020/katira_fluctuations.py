from math import degrees, atan

import cv2
import numpy as np
from scipy import spatial
from scipy.ndimage import interpolation

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

    # Get the dh's for each frame
    dh = [get_dh(data.df.loc[f], ims[f]) for f in range(2)]
    print(dh)


def get_dh(data, im):
    cgw = get_cgw(data)
    field, x, y = coarse_order_field(data, cgw)
    phi_d, phi_o = get_phis(field)
    criteria = (phi_d + phi_o) / 1.6
    binary = get_binary_image(field, criteria)

    # Find the angle of the image
    angle = get_im_angle(im)

    # Rotate the image then dilate it
    binary = interpolation.rotate(binary, angle)
    binary = np.uint8(binary)
    binary = images.dilate(binary, (5, 5))

    # Filter the image
    binary = filter_by_contours(binary)

    # Select border
    xmin, xmax, h = get_border(interpolation.rotate(im, angle))

    # Find closest points to boundary on binary image
    x, y = get_boundary(binary, xmin, xmax, h)

    # Find dh
    dh = np.array(y) - h
    return dh


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


def get_border(im):
    ls = LineSelector(im)
    return ls.points[0][0], ls.points[1][0], ls.points[0][1]


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
    return angle


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


def get_cgw(data):
    tree = spatial.cKDTree(data[['x', 'y']].values)
    dists, indices = tree.query(tree.data, 2)
    cgw = np.mean(dists[1, :])
    return cgw


def coarse_order_field(df, cgw=30):
    points = df[['x', 'y']].values
    order = df.order.values
    x = np.arange(0, max(df.x), 1)
    y = np.arange(0, max(df.y), 1)
    x, y = np.meshgrid(x, y)
    r = np.dstack((x, y))

    points_for_tree = df.loc[df.order > 0.8, ['x', 'y']].values
    tree = spatial.cKDTree(points_for_tree)
    tree_dists, tree_indxs = tree.query(r, 20)
    cgw = np.mean(tree_dists[:, :, 0])
    print(cgw)
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
    run_analysis(files, savename)

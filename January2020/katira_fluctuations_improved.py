from math import pi

import cv2
import numpy as np
from scipy import spatial
from tqdm import tqdm

from Generic import images, filedialogs
from ParticleTracking import dataframes, statistics


def run(files, savename, lattice_spacing=5):
    N = len(files)
    ims = [images.load(f, 0) for f in files]

    # Find circles in all images
    circles = [images.find_circles(im, 27, 200, 7, 16, 16) for im in
               tqdm(ims, 'Finding circles')]

    # Save data to a dataframe
    data = dataframes.DataStore(savename, load=False)
    for f, info in enumerate(circles):
        data.add_tracking_data(f, info, ['x', 'y', 'r'])

    # Calculate order parameter
    calc = statistics.PropertyCalculator(data)
    calc.order()

    # Calculate cgw, the average spacing between particles in the ordered phase
    cgw = get_cgw(data.df.loc[0]) / 2

    # Calculate the coarse order field for all frames
    fields = []
    for f in tqdm(range(N), 'Calculating fields'):
        field, x, y = coarse_order_field(data.df.loc[f], cgw, lattice_spacing)
        fields.append(field)

    field_threshold = get_field_threshold(fields, lattice_spacing, ims[0])

    # Interpolate the lattice nodes in the fields to find the set of points
    # that satisfy the field threshold


def get_field_threshold(fields, ls, im):
    # Draw a box around an always ordered region of the image to
    # calculate the phi_o
    fields = np.array(fields)
    line_selector = LineSelector(im)
    op1, op2 = line_selector.points
    phi_o = np.mean(
        fields[:, op1[1] // ls:op2[1] // ls, op1[0] // ls:op2[0] // ls])

    # Repeat for disordered
    fields = np.array(fields)
    line_selector = LineSelector(im)
    dp1, dp2 = line_selector.points
    phi_d = np.mean(
        fields[:, dp1[1] // ls:dp2[1] // ls, dp1[0] // ls:dp2[0] // ls])

    field_threshold = (phi_o + phi_d) / 2
    return field_threshold


def get_cgw(df):
    tree = spatial.cKDTree(df[['x', 'y']].values)
    dists, _ = tree.query(tree.data, 2)
    cgw = np.mean(dists[:, 1])
    return cgw


def coarse_order_field(df, cgw, node_spacing=1, no_of_neighbours=20):
    """
    Calculate the coarse-grained field characterising local orientation order
    """

    order = df.order.values

    # Generate the lattice nodes to query
    x = np.arange(0, max(df.x), node_spacing)
    y = np.arange(0, max(df.y), node_spacing)
    x, y = np.meshgrid(x, y)
    r = np.dstack((x, y))

    # Get the positions of all the particles
    particles = df[['x', 'y']].values

    # Generate the tree from the particles
    tree = spatial.cKDTree(particles)

    # Query the tree at all the lattice nodes to find the nearest n particles
    # Set n_jobs=-1 to use all cores
    dists, indices = tree.query(r, no_of_neighbours, n_jobs=-1)

    # Calculate all the coarse-grained delta functions (Katira ArXiv eqn 3
    cg_deltas = np.exp(-dists ** 2 / (2 * cgw ** 2)) / (2 * pi * cgw ** 2)

    # Multiply by the orders to get the summands
    summands = cg_deltas * order[indices]

    # Sum along axis 2 to calculate the field
    field = np.sum(summands, axis=2)

    return field, x, y


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
    direc = "/media/data/Data/January2020/RecordFluctuatingInterface/Quick/first_frames"
    files = filedialogs.get_files_directory(direc + '/*.png')
    savename = direc + '/data.hdf5'
    run(files, savename)

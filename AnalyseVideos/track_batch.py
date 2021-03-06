import os
import warnings

from Generic import filedialogs
from ParticleTracking.tracking.tp_locate_nitrile import TrackpyPT

warnings.filterwarnings('ignore')

directory = filedialogs.open_directory('Open Directory containing videos')
files = os.listdir(directory)

for i, file in enumerate(files):
    print(i + 1, 'of', len(files))
    file = directory + '/' + file
    name, ext = os.path.splitext(file)
    if ext == '.MP4':
        print(file)
        data_file = name + '.hdf5'
        if not os.path.exists(data_file):
            tracker = TrackpyPT(file, tracking=True, multiprocess=True)
            tracker.track()
            del tracker

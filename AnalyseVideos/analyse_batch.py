import warnings
import os

from Generic import filedialogs
from ParticleTracking import dataframes, statistics, annotation
from ParticleTracking.tracking.james_nitrile import JamesPT

warnings.filterwarnings('ignore')

directory = filedialogs.open_directory('Open Directory containing videos')
files = os.listdir(directory)

for file in files:
    file = directory + '/' + file
    name, ext = os.path.splitext(file)
    if ext == '.hdf5':
        print(ext)
        continue
    data_file = name + '.hdf5'
    if not os.path.exists(data_file):
        tracker = JamesPT(file, tracking=True, multiprocess=True)
        tracker.track()
        del tracker
    data_store = dataframes.DataStore(file, load=True)
    # annotation.CircleAnnotator(file, data_store, 'particle').annotate()
    calculator = statistics.PropertyCalculator(data_store)
    calculator.order(multiprocessing=True)
    del data_store, calculator

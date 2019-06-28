import warnings

from Generic import filedialogs
from ParticleTracking import dataframes, statistics

warnings.filterwarnings('ignore')

directory = filedialogs.open_directory('Open Directory containing videos')
# files = os.listdir(directory+'/*.hdf5')
files = filedialogs.get_files_directory(directory + '/*.hdf5')

for file in files:
    with dataframes.DataStore(file) as data:
        # annotation.CircleAnnotator(file, data, 'particle').annotate()
        calculator = statistics.PropertyCalculator(data)
        calculator.order()
        calculator.density()

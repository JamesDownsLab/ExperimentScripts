import warnings
import time

from Generic import filedialogs
from ParticleTracking import dataframes, statistics
from ParticleTracking.tracking.james_nitrile import JamesPT
from ParticleTracking.tracking.example_child import ExampleChild
from ParticleTracking import annotation

warnings.filterwarnings("ignore")

### Load a file ###
###################
file = filedialogs.load_filename('Load a video', remove_ext=False, directory='/home/ppxjd3/Videos/')
# file = "/home/ppxjd3/Videos/test_move.MP4"

### Tracking ###
###############

# tracker = JamesPT(file, tracking=True, multiprocess=False)
# ExampleChild(file, tracking=True).track()
# tracker.track()

#
data_store = dataframes.DataStore(file, load=True)
data_name = data_store.filename
# data_store.inspect_dataframes()

### Annotations ###
###################
# annotation.CircleAnnotator(file, data_store, 'real order', False).annotate()
# # # annotation.neighbors(data_store, 0)
# annotator = video.CircleAnnotate(data_store, file)

### Statistics ###
##################
# data_store = None
calculator = statistics.PropertyCalculator(data_store)
# print(data_store.num_frames)
# calculator.distance(multiprocess=True)
# calculator.level_checks()
# t = time.time()
calculator.order(multiprocessing=True)
# calculator.test()
# calculator.density(multiprocess=True)
# calculator.correlations(1, r_min=1, r_max=20, dr=0.04)

### Graphs ###
##############
# from ExperimentScripts.AnalyseData.Correlations import plot_correlations
# plot_correlations.corr(file, 1)


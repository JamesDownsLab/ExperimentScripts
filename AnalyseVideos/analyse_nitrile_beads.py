from ParticleTracking import dataframes, annotation
from ParticleTracking.tracking_methods.james_nitrile import JamesPT
from Generic import filedialogs
import warnings
warnings.filterwarnings("ignore")

### Load a file ###
###################
file = filedialogs.load_filename('Load a video', remove_ext=False, directory='/home/ppxjd3/Videos/')
# file = "/home/ppxjd3/Videos/test_move.MP4"

### Tracking ###
###############

tracker = JamesPT(file, tracking=True, multiprocess=False)
tracker.track()

#
data_store = dataframes.DataStore(file, load=True)
# data_store.inspect_dataframes()

### Annotations ###
###################
annotator = annotation.VideoAnnotator(data_store, file)
annotator.add_coloured_circles()
# # # annotation.neighbors(data_store, 0)
# annotator = video.CircleAnnotate(data_store, file)

### Statistics ###
##################
# calculator = statistics.PropertyCalculator(data_store)
# print(data_store.num_frames)
# calculator.distance()
# calculator.level_checks()
# calculator.order()
# calculator.correlations(1, r_min=1, r_max=20, dr=0.04)

### Graphs ###
##############
# from ExperimentScripts.AnalyseData.Correlations import plot_correlations
# plot_correlations.corr(file, 1)


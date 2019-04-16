from ParticleTracking import dataframes, configurations, tracking, statistics, annotation
from Generic import filedialogs, video
import warnings
warnings.filterwarnings("ignore")
import time

### Load a file ###
###################
file = filedialogs.load_filename('Load a video', remove_ext=False, directory='/home/ppxjd3/Videos/')
# file = "/home/ppxjd3/Videos/test_move.MP4"
### Tracking ###
###############
options = configurations.NITRILE_BEADS_PARAMETERS

pt = tracking.ParticleTracker(file, options, multiprocess=False,
                              crop_method='auto', show_debug=False)
# s = time.time()
pt.track()
# print(time.time() - s)
#
data_store = dataframes.DataStore(file, load=True)
data_store.inspect_dataframes()

### Annotations ###
###################
# annotator = annotation.VideoAnnotator(data_store, file)
# annotator.add_coloured_circles()
# # # annotation.neighbors(data_store, 0)
annotator = video.CircleAnnotate(data_store, file)

### Statistics ###
##################
# calculator = statistics.PropertyCalculator(data_store)
# calculator.distance()
# calculator.edge_distance()
# calculator.level_checks()
# calculator.order()
# calculator.susceptibility()
# calculator.density()
# calculator.average_density()
# calculator.correlations(300, r_min=1, r_max=20, dr=0.04)
# calculator.correlations(10)

### Graphs ###
##############

# graphs.order_quiver(data_store, 0)

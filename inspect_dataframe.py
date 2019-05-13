from ParticleTracking import dataframes
from Generic import plotting, filedialogs

file = filedialogs.load_filename(directory='/media/data/Data',
                                 file_filter='*.hdf5')
data = dataframes.DataStore(file)


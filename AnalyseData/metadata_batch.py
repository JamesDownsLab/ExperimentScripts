from Generic import filedialogs
from ParticleTracking import dataframes

directory = filedialogs.open_directory('Open Directory containing videos')
files = filedialogs.get_files_directory(directory + '/*.hdf5')

n = []
for file in files:
    meta = dataframes.load_metadata(file)
    print(meta)
    n.append(meta['number_of_particles'])

from Generic import filedialogs
from ParticleTracking import correlations

directory = filedialogs.open_directory('Open Directory containing videos')
files = filedialogs.get_files_directory(directory + '/*.hdf5')

for file in files:
    if 'corr' not in file:
        print(file)
        correlations.calculate_corr_data(file)

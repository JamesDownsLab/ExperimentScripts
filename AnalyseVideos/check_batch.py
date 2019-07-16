from Generic import filedialogs
from ParticleTracking import check

directory = filedialogs.open_directory()
files = filedialogs.get_files_directory(directory + '/*.hdf5')

for file in files:
    check.tracking(file)

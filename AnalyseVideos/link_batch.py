from Generic import filedialogs
from ParticleTracking import linking, configurations

directory = filedialogs.open_directory('')

files = filedialogs.get_files_directory(directory+'/*.hdf5')
config = configurations.TRACKPY_NITRILE_PARAMETERS

for file in files:
    linker = linking.Linker(file)
    linker.link(config['search_range'], memory=config['memory'])
    linker.quit()
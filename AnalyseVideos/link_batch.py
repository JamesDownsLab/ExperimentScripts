from Generic import filedialogs
from ParticleTracking import linking, configurations
import logging

ch = logging.StreamHandler()
formatter = logging.Formatter('\x1b[80D\x1b[1A\x1b[K%(message)s')
ch.setFormatter(formatter)

directory = filedialogs.open_directory('')

files = filedialogs.get_files_directory(directory+'/*.hdf5')
config = configurations.TRACKPY_NITRILE_PARAMETERS

for file in files:
    print(file)
    linker = linking.Linker(file)
    linker.link(config['search_range'], memory=config['memory'])
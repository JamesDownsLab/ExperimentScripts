from ParticleTracking.tracking import james_nitrile as jn
from ParticleTracking import dataframes
import os
import numpy as np


def add_duty(file):
    data_name = os.path.splitext(file)[0] + '.hdf5'
    vid_name = os.path.splitext(file)[0] + '.MP4'
    data = dataframes.DataStore(data_name, load=True)
    print(np.unique(data.frame_data.Duty.values))
    duty_cycle = jn.read_audio_file(vid_name, data.num_frames)
    data.add_frame_property('Duty', duty_cycle)
    data.save()

if __name__ == "__main__":
    from Generic import filedialogs
    file = filedialogs.load_filename(directory='/media/data/Data')
    add_duty(file)

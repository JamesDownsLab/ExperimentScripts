from ParticleTracking.tracking import james_nitrile as jn
from Generic import video, filedialogs
import matplotlib.pyplot as plt
import numpy as np

file = filedialogs.load_filename(directory="/media/data/Data")
vid = video.ReadVideo(file)
frames = vid.num_frames
freqs = jn.read_audio_file(file, frames)
d = np.round((freqs - 1000)/15)
fig, ax = plt.subplots()
frames = list(range(frames))
ax.plot(frames, d)
ax.set_xlabel('frame')
ax.set_ylabel('Duty Cycle / 1000')
ax2 = ax.twinx()
ax2.plot(frames, d)
ax2.set_ylabel('Frequency (Hz)')
plt.show()
from ParticleTracking import tracking
from Generic import video
import matplotlib.pyplot as plt

file = "/media/ppxjd3/CAM_SD1/DCIM/205RBRHM/22050002.MP4"
vid = video.ReadVideo(file)
frames = vid.num_frames
d = tracking.read_audio_file(file, frames)
plt.figure()
plt.plot(d)
plt.show()
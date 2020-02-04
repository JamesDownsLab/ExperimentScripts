import os

from tqdm import tqdm

from Generic import images, video, filedialogs

direc = filedialogs.open_directory()
files = filedialogs.get_files_directory(direc + '/*.MP4')

if not os.path.exists(direc + '/first_frames/'):
    os.mkdir(direc + '/first_frames')

i = True
for file in tqdm(files):
    folder, name = os.path.split(file)
    name = name.split('.')[0]
    save_name = folder + '/first_frames/' + name + '.png'
    with video.ReadVideo(file) as vid:
        frame = vid.read_next_frame()
        if i:
            i = False
            cropper = images.InteractiveCrop(frame, 6)
            mask, crop, _, _ = cropper.begin_crop()
        frame = images.crop_and_mask_image(frame, crop, mask)
        images.save(frame, save_name)

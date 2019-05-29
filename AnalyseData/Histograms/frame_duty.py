from ParticleTracking import dataframes

def frame_duty(file):
    data = dataframes.DataStore(file, load=True)
    print(data.frame_data.head())


if __name__ == "__main__":
    from Generic import filedialogs
    file = filedialogs.load_filename()
    frame_duty(file)
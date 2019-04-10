from Shaker import arduino
import numpy as np
import time


SPEAKER = "/dev/serial/by-id/usb-Arduino_LLC_Arduino_Micro-if00"
speaker = arduino.Arduino(port=SPEAKER, rate=115200, wait=False)
rate = 10

duty_cycles = np.arange(0, 10000, 100)
for d in duty_cycles:
    time.sleep(1/rate)
    string = 'i{:03}'.format(d)
    speaker.send_serial_line(string[1:])


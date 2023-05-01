import MicStreaming
import numpy as np
import time
import threading

# Initialize Microphone
mic1 = MicStreaming.Streaming()
mic1.device_index = 1
mic1.start_recording()

thread1 = threading.Thread(target=mic1.stream_read)
thread1.start()
time.sleep(2)
mic1.magnitude()

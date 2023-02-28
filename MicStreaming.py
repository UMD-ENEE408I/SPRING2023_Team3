import pyaudio
import wave
import sys
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fs = 44100 # samples per second
FORMAT = pyaudio.paInt16 # 16 bits per sample
CHANNELS = 1 # Audio Channels
chunk = 1024 # record in chunks of 1024 samples
filename = "output.wav"

#def callback(in_data, frame_count, time_info, status):
 #   data = np.frombuffer(in_data, dtype=np.Int16)
  #  print(data)



audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=fs,
                    input=True,
                    frames_per_buffer=chunk)

frames = [] # stores recorded data
# Store data in chunks for 3 seconds
for i in range(0, int(fs/ chunk * 10)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

wf = wave.open(filename, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

spf = wave.open("output.wav", "r")
signal = spf.readframes(-1)
signal = np.frombuffer(signal, dtype=np.int16)
fs = spf.getframerate()
Time = np.linspace(0, len(signal) / fs, num=len(signal))

plt.figure(1)
plt.title("Signal Wave...")
plt.plot(Time, signal)
plt.show()
# Start animation

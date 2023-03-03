import pyaudio
import sys
import time
import wave
import matplotlib.pyplot as plt
import numpy as np

fs = 44100  # samples per second
FORMAT = pyaudio.paInt16  # 16 bits per sample
CHANNELS = 1  # Audio Channels
chunk = 1024  # record in chunks of 1024 samples
frames = []  # stores recorded data
save_length = 7.5  # Length of audio data saved from end
start_time = time.time()
filename = "output.wav"

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=fs,
                    input=True,
                    frames_per_buffer=chunk)


while time.time() - start_time < 10:
    # Get Audio Data into frame. runs for time interval
    data = np.frombuffer(stream.read(chunk), dtype=np.int16)
    frames.append(data)
    # Only keep the last 300 chunks (7.5 seconds)
    frames = frames[-int(((fs / chunk) * save_length)):]
    # plot frames

amplitude = np.hstack(frames)
Time = np.linspace(0, len(amplitude) / fs, num=len(amplitude))
plt.plot(Time, amplitude)
plt.show()

stream.stop_stream()
stream.close()
audio.terminate()

wf = wave.open(filename, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()
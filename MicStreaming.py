import pyaudio
import wave
import sys

sampleRate = 44100 # samples per second
FORMAT = pyaudio.paInt16 # 16 bits per sample
CHANNELS = 1
chunk = 1024 # record in chunks of 1024 samples
filename = "output.wav"

audio = pyaudio.PyAudio()
# Test
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=sampleRate, input=True, frames_per_buffer=chunk)

frames = [] # stores recorded data
# Store data in chunks for 3 seconds
for i in range(0, int(sampleRate/ chunk * 10)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

wf = wave.open(filename, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(sampleRate)
wf.writeframes(b''.join(frames))
wf.close()
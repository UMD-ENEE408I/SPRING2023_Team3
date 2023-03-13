import pyaudio
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

class Streaming(object):
    def __init__(self):
        print("Initializing Microphone...")
        self.fs = 44100  # samples per second
        self.FORMAT = pyaudio.paInt16  # 16 bits per sample
        self.CHANNELS = 1  # Audio Channels
        self.chunk = 1024  # record in chunks of 1024 samples
        self.save_length = 5  # Length of audio data saved from end (s)
        self.t = []  # stores time corresponding to each sample
        self.frames = []  # stores recorded data
        self.start_time = time.time()
        self.audio = pyaudio.PyAudio()

    def start_recording(self):
        self.stream = self.audio.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.fs,
                            input=True,
                            frames_per_buffer=self.chunk,
                            input_device_index=1)
        print("Started Audio Stream")

    # Need to add methods to get the time and data vector of the recording
    def stream_read(self):
        try:
            while True:
                data = np.frombuffer(self.stream.read(self.chunk), dtype=np.int16)
                t = self.stream.get_time()
                self.frames.append(data)
                # Only keep the last 300 chunks (5 seconds)
                self.frames = self.frames[-int(((self.fs / self.chunk) * self.save_length)):]
                self.t = self.t[-len(self.frames):]
        except:
            print("Exception")
            return

    def plot_data(self):
        amplitude = np.hstack(self.frames)
        x = np.linspace(0, len(amplitude) / self.fs, num=len(amplitude))
        plt.plot(x, amplitude)
        plt.show()


    def stop_recording(self):
        if 'stream' in locals():
            self.stream.stop_stream()
            self.stream.close()
        print("Recording Stopped")

    def close(self):
        self.stream_stop()
        self.audio.terminate()

if __name__=="__main__":
    mic1 = Streaming()
    mic1.start_recording()
    mic1.stream_read()
    mic1.plot_data()
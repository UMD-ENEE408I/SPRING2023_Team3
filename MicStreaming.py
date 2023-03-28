import pyaudio
import threading
import time
import matplotlib.pyplot as plt
import numpy as np


class Streaming(object):
    def __init__(self):
        print("Initializing Microphone...")
        self.fs = 44100  # samples per second
        self.FORMAT = pyaudio.paInt16  # 16 bits per sample
        self.CHANNELS = 1  # Audio Channels
        self.chunk = 1024  # record in chunks of 1024 samples
        self.save_length = .15  # Length of audio data saved from end (s)
        self.t = []  # stores time corresponding to each sample
        self.frames = []  # stores recorded data
        self.start_time = time.time()
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.tv = []
        self.stream_open = False
        self.audio_data = []


    def start_recording(self):
        self.stream_open = True
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.fs,
                                      input=True,
                                      frames_per_buffer=self.chunk,
                                      input_device_index=1)
        print("Started Audio Stream")

    # Need to add methods to get the time and data vector of the recording
    def stream_read(self):
        print("Recording...")
        try:
            while True:
                data = np.frombuffer(self.stream.read(self.chunk), dtype=np.int16)
                self.t.append(self.stream.get_time())
                self.frames.append(data)
                # Only keep the last 300 chunks (5 seconds)
                self.frames = self.frames[-int(((self.fs / self.chunk) * self.save_length)):]
                self.t = self.t[-len(self.frames):]
                self.stream_open = True
        except:
            print("Stream Stopped: Exception")
            return

    def plot_data(self):
        amplitude = np.hstack(self.frames)
        x = np.linspace(0, len(amplitude) / self.fs, num=len(amplitude))
        plt.plot(x, amplitude)
        plt.show()

    def plot_two(self, mic2):
        a1 = np.hstack(self.frames)
        a2 = np.hstack(mic2.frames)
        self.create_space()
        mic2.create_space()
        figure, axis = plt.subplots(1,2)
        axis[0].plot(self.tv, a1)
        axis[1].plot(mic2.tv, a2)
        plt.show()

    def stop_recording(self):
        if 'stream' in locals():
            self.stream.stop_stream()
            self.stream.close()
            self.stream_open = False
        print("Recording Stopped")

    def close(self):
        self.stop_recording()
        self.audio.terminate()

    def create_space(self):
        t_data = self.t
        raw_data = self.frames
        # Create a linspace of the data with time values for each sample
        first = t_data[0]
        last = t_data[-1]
        self.audio_data = np.hstack(raw_data)
        self.tv = np.linspace(first, last, num=len(self.audio_data))


    def sync_time(self, mic2):
        if (time.time() - self.start_time) > 1:
            time_diff = mic2.tv[0] - self.tv[0]
            if(np.abs(time_diff) > 1/self.fs):
                # Samples every 2.2498e-5
                # Shift array to match
                shift = round(time_diff / (1/self.fs))
                if time_diff < 0: # Mic1 starts after mic2
                    mic2.tv = mic2.tv[shift:]
                    self.tv = self.tv[:-shift]
                    mic2.audio_data = mic2.audio_data[shift:]
                    self.audio_data = self.audio_data[:-shift]
                if time_diff > 0: # Mic2 starts after Mic1
                    mic2.tv = mic2.tv[:-shift]
                    self.tv = self.tv[shift:]
                    mic2.audio_data = mic2.audio_data[:-shift]
                    self.audio_data = self.audio_data[shift:]


    def cross_correlation(self, mic2):
        if (time.time() - self.start_time) > 2:
            self.create_space()
            mic2.create_space()
            td = self.tv[0] - mic2.tv[0]
            #self.sync_time(mic2)
            corr = np.correlate(self.audio_data, mic2.audio_data, mode='full')
            delay = np.argmax(corr)
            t_delay = delay * (1 / self.fs) + td
            return t_delay
        # Need to make arrays much smaller. Takes forever to compute the cross correlation.


    def time_delay(self, mic2):
        while mic1.stream_open and mic2.stream_open:
            tdoa = mic1.cross_correlation(mic2)
            print(tdoa)

if __name__ == "__main__":
    mic1 = Streaming()
    mic1.start_recording()
    mic2 = Streaming()
    mic2.start_recording()
    thread1 = threading.Thread(target=mic1.stream_read)
    thread2 = threading.Thread(target=mic2.stream_read)
    thread1.start()
    thread2.start()
    #mic1.time_delay(mic2)
    time.sleep(5)
    mic1.plot_two(mic2)







    
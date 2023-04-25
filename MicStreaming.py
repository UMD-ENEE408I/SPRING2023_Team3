import pyaudio
import threading
import time
import matplotlib.pyplot as plt
import numpy as np


import filter
import sounddevice as sd
from scipy.fft import fft, ifft, fftfreq
from scipy import signal


class Streaming(object):
    def __init__(self):
        print("Initializing Microphone...")
        self.fs = 44100  # samples per second
        self.FORMAT = pyaudio.paInt16  # 16 bits per sample
        self.CHANNELS = 1  # Audio Channels
        self.chunk = 1024  # record in chunks of 1024 samples
        self.device_index = None # Specify input device
        self.save_length = .25  # Length of audio data saved from end (s)
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
                                      input_device_index=self.device_index)
        print("Started Audio Stream")

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
        if time.time() - self.start_time > 1:
            t_data = self.t
            raw_data = self.frames
            # Create a linspace of the data with time values for each sample
            first = t_data[0]
            last = t_data[-1]
            self.audio_data = np.hstack(raw_data)
            self.tv = np.linspace(first, last, num=len(self.audio_data))

    def sync_time(self, mic2):
        self.create_space()
        mic2.create_space()
        time1_origin = self.tv[0]
        time2_origin = mic2.tv[0]
        diff = time2_origin - time1_origin
        index_shift = abs(int(diff/ (1/self.fs)))
        if diff > 0:
            # Mic2 starts after mic1 by index_shift samples
            self.tv = self.tv[index_shift:]
            self.audio_data = self.audio_data[index_shift:]
            mic2.tv = mic2.tv[:-index_shift]
            mic2.audio_data = self.audio_data[:-index_shift]
        else:
            # Mic1 starts after Mic2 by index_shift samples
            self.tv = self.tv[:-index_shift]
            self.audio_data = self.audio_data[:-index_shift]
            mic2.tv = mic2.tv[index_shift:]
            mic2.audio_data = mic2.audio_data[index_shift:]



    def filter(self, filter_freq, plot=False):
        lowBound = (filter_freq - 100)
        highBound = (filter_freq + 100)

        self.audio_data = filter.normalize(self.audio_data)
        transform = fft(self.audio_data)

        # sos = signal.butter(4, [lowBound, highBound], btype='bandpass')
        N = len(transform)
        xf = fftfreq(transform.size, 1 / self.fs)
        fTransform = np.copy(transform)
        fTransform[np.abs(xf) <= lowBound] = 0
        fTransform[np.abs(xf) >= highBound] = 0

        filteredAmplitude = ifft(fTransform)
        filteredAmplitude = filter.normalize(filteredAmplitude)
        fn = len(filteredAmplitude)
        # amplitude = np.hstack(self.frames)
        amplitude = self.audio_data
        # Plot Raw audio and FFT of both filtered and unfiltered amplitude
        if plot:
            xt = np.linspace(0, len(amplitude) / self.fs, num=len(amplitude))
            fig, ax = plt.subplots(1, 4)
            ax[0].plot(xf[:(N - 1)], transform[:(N - 1)], 'r')
            ax[1].plot(xt, amplitude,  'b')
            ax[2].plot(xf[:(N - 1)], fTransform[:(N - 1)], 'r')
            ax[3].plot(xt, filteredAmplitude, 'g')
            plt.show()

        return filteredAmplitude


    def cross_correlation(self, mic2, freq):
        self.sync_time(mic2)
        fa1 = self.filter(freq, plot=True)
        fa2 = mic2.filter(freq, plot=True)

        corr = signal.correlate(fa1, fa2, mode='full')
        C_norm1 = np.zeros(corr.shape[0])
        C_norm2 = np.zeros(corr.shape[0])
        N = len(fa1)
        step = 1 / self.fs
        t_shift_C = np.arange((- (N * step)) + step, N * step, step)
        center_index = int((corr.shape[0] + 1) / 2) - 1  # Index corresponding to zero shift
        low_shift_index = -int((corr.shape[0] + 1) / 2) + 1
        high_shift_index = int((corr.shape[0] + 1) / 2) - 1

        for i in range(low_shift_index, high_shift_index + 1):
            low_norm_index = max(0, i)
            high_norm_index = min(fa1.shape[0], i + fa1.shape[0])
            C_norm1[i + center_index] = np.linalg.norm(fa1[low_norm_index:high_norm_index])

            low_norm_index = max(0, -i)
            high_norm_index = min(fa2.shape[0], -i + fa2.shape[0])
            C_norm2[i + center_index] = np.linalg.norm(fa2[low_norm_index:high_norm_index])

        corr_normalized = corr / (C_norm1 * C_norm2)
        max_indices_back = -int(((1 / 10) / 2) / (1 / mic1.fs)) + center_index
        max_indices_forward = int(((1 / 10) / 2) / (1 / mic1.fs)) + center_index
        i_max_C_normalized = np.argmax(corr_normalized)
        i_max_C_normalized = np.argmax(corr_normalized[max_indices_back:max_indices_forward + 1]) + max_indices_back
        t_shift_hat_normalized = t_shift_C[i_max_C_normalized]
        ind = len(corr_normalized)

        plt.plot(t_shift_C[:ind], corr_normalized)
        plt.show()

        return t_shift_hat_normalized


    def time_delay(self, mic2, freq):
        while self.stream_open and mic2.stream_open:
            delay = mic1.cross_correlation(mic2, freq)
            tdoa = abs(delay)
            print('Time Delay: ', tdoa)
            print('Distance (m):', tdoa * 343)
            time.sleep(.25)


if __name__ == "__main__":
    mic1 = Streaming()
    mic1.device_index = 1
    mic1.start_recording()
    mic2 = Streaming()
    mic2.device_index = 2
    mic2.start_recording()
    thread1 = threading.Thread(target=mic1.stream_read)
    thread2 = threading.Thread(target=mic2.stream_read)
    thread1.start()
    thread2.start()
    time.sleep(2)
    D = mic1.time_delay(mic2, 4000)
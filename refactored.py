import os
import wave
import time
import pickle
import pyaudio
import warnings
import random
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
import time
from tkinter import *
from tkinter import messagebox
from itertools import count

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
DURATION_SECONDS = 5
NO_TEST_SAMPLES = 2
PASSWORDS = list(open('passwords.txt'))
LOCALPATH = "C:\\Users\\artes\\OneDrive\\Semestr 9\\SM\\Projekt\\"
SAMPLE_NUMBER_FOR_ONE_MODEL = 0


class Identification:
    def __init__(self):
        self.ws_main = self.initialize_window("Identyfikacja")
        self.recording_device = IntVar()
        self.pyAudio = pyaudio.PyAudio()
        self.initialize_navigation()
        self.initialize_main_view()
        self.ws_main.mainloop()

    def initialize_window(self, title):
        ws = Tk()
        ws.geometry("700x350")
        ws.title(title)

        return ws

    def initialize_navigation(self):
        frame = Frame(self.ws_main)
        frame.pack()
        mainmenu = Menu(frame)
        mainmenu.add_command(label="Dodaj dane", command=self.nav_add_model)
        mainmenu.add_command(label="Trenuj model",
                             command=self.nav_train_models)
        mainmenu.add_command(
            label="Identifikuj nagrane próbki", command=self.nav_identify_samples)
        mainmenu.add_command(label="Exit", command=self.ws_main.destroy)
        self.ws_main.config(menu=mainmenu)

    def initialize_main_view(self):
        self.initialize_recording_devices()
        label_title = Label(
            self.ws_main, text="Identyfikacja użytkownika", font='Arial 18')
        label_title.place(relx=0.5, rely=0.10, anchor=CENTER)

        label_info = Label(
            self.ws_main, text="Wciśnij przycisk i wypowiedź wyświetlone poniżej hasło:", font='Arial 15')
        label_info.place(relx=0.5, rely=0.4, anchor=CENTER)

        label_password = Label(self.ws_main, text=random.choice(
            PASSWORDS), font='Arial 17 bold')
        label_password.place(relx=0.5, rely=0.6, anchor=CENTER)

        start_btn = Button(
            self.ws_main,
            text='START',
            bd='5',
            command=self.record_sample
        )

        start_btn.place(relx=0.5, rely=0.7, anchor=CENTER)

    def initialize_recording_devices(self):
        recordingDevicesList = self.get_recording_devices()

        # Set recording device manually
        # self.recording_device.set(1)

        for i in range(len(recordingDevicesList)):
            Radiobutton(self.ws_main,
                        text=recordingDevicesList[i],
                        padx=20,
                        variable=self.recording_device,
                        value=i).pack(side="bottom")

    def get_recording_devices(self):
        pyAudio = pyaudio.PyAudio()
        list_of_devices = []
        data = pyAudio.get_host_api_info_by_index(0)
        for i in range(0, data.get('deviceCount')):
            if (pyAudio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                list_of_devices.append(
                    pyAudio.get_device_info_by_host_api_device_index(0, i).get('name'))

        return list_of_devices

    def calculate_delta(self, array):
        rows, cols = array.shape
        print(rows)
        print(cols)
        deltas = np.zeros((rows, 20))
        N = 2
        for i in range(rows):
            index = []
            j = 1
            while j <= N:
                if i-j < 0:
                    first = 0
                else:
                    first = i-j
                if i+j > rows-1:
                    second = rows-1
                else:
                    second = i+j
                index.append((second, first))
                j += 1
            deltas[i] = (array[index[0][0]]-array[index[0][1]] +
                         (2 * (array[index[1][0]]-array[index[1][1]]))) / 10
        return deltas

    def extract_features(self, audio, rate):
        mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01,
                                 20, nfft=1200, appendEnergy=True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        print(mfcc_feature)
        delta = self.calculate_delta(mfcc_feature)
        combined = np.hstack((mfcc_feature, delta))
        return combined

    def add_model(self, ws_add_model, modelName, label_password, label_record):
        global SAMPLE_NUMBER_FOR_ONE_MODEL

        stream = self.pyAudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.recording_device.get(),
            frames_per_buffer=CHUNK
        )

        label_password.place_forget()
        label_password = Label(
            ws_add_model, text=PASSWORDS[SAMPLE_NUMBER_FOR_ONE_MODEL], font='Arial 17 bold')
        label_password.place(relx=0.5, rely=0.6, anchor=CENTER)

        label_record = Label(ws_add_model, text="Nagrywam...", font='Arial 15')
        label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
        ws_add_model.update()

        frames = []
        for i in range(0, int(RATE / CHUNK * DURATION_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        label_record.place_forget()
        label_record = Label(
            ws_add_model, text="Koniec nagrania...", font='Arial 15')
        label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
        stream.stop_stream()
        stream.close()
        self.pyAudio.terminate()

        OUTPUT_FILENAME = modelName + "-sample" + \
            str(SAMPLE_NUMBER_FOR_ONE_MODEL) + ".wav"
        WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)
        trainedfilelist = open("training_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME+"\n")
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(self.pyAudio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        time.sleep(1)
        label_record.place_forget()
        label_password.place_forget()
        SAMPLE_NUMBER_FOR_ONE_MODEL += 1

    def nav_add_model(self):
        ws_add_model = self.initialize_window("Nowy model")

        label_title = Label(
            ws_add_model, text="Dodawanie profilu użytkownika", font='Arial 18')
        label_title.place(relx=0.5, rely=0.10, anchor=CENTER)

        modelName = StringVar()

        name_label = Label(ws_add_model, text='Nazwa użytkownika',
                           font=('calibre', 10, 'bold'))
        name_entry = Entry(ws_add_model, textvariable=modelName,
                           font=('calibre', 10, 'normal'))
        name_label.place(relx=0.4, rely=0.3, anchor=CENTER)
        name_entry.place(relx=0.6, rely=0.3, anchor=CENTER)

        label_info = Label(
            ws_add_model, text="Wciśnij przycisk i wypowiedź wyświetlone poniżej hasło:", font='Arial 15')
        label_info.place(relx=0.5, rely=0.4, anchor=CENTER)

        label_password = Label(
            ws_add_model, text=PASSWORDS[0], font='Arial 17 bold')
        label_password.place(relx=0.5, rely=0.6, anchor=CENTER)

        start_btn = Button(
            ws_add_model,
            text='START',
            bd='5',
            command=lambda: self.add_model(
                ws_add_model, name_entry.get(), label_password, label_info)
        )

        start_btn.place(relx=0.5, rely=0.7, anchor=CENTER)

    def nav_train_models(self):
        source = LOCALPATH + "training_set\\"
        dest = LOCALPATH + "trained_models\\"
        train_file = LOCALPATH + "training_set_addition.txt"
        file_paths = open(train_file, 'r')
        count = 1
        features = np.asarray(())
        for path in file_paths:
            path = path.strip()
            print(path)

            sr, audio = read(source + path)
            print(sr)
            vector = self.extract_features(audio, sr)

            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))

            if count == 5:
                gmm = GaussianMixture(
                    n_components=6, max_iter=200, covariance_type='diag', n_init=3)
                gmm.fit(features)

                # dumping the trained gaussian model
                picklefile = path.split("-")[0]+".gmm"
                pickle.dump(gmm, open(dest + picklefile, 'wb'))
                print('+ modeling completed for speaker:', picklefile,
                      " with data point = ", features.shape)
                features = np.asarray(())
                count = 0
            count = count + 1

    def nav_identify_samples(self):
        print('here will identify samples')
        return

    def record_sample(self):
        print('here will record testing sample')
        return


Identification()

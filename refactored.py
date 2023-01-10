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


class Identification:
    def __init__(self):
        self.ws = self.initialize_window()
        self.recording_device = IntVar()
        self.initialize_navigation()
        self.initialize_main_view()
        self.ws.mainloop()

    def initialize_window(self):
        ws = Tk()
        ws.geometry("700x350")
        ws.title("Identyfikacja użytkownika")

        return ws

    def initialize_navigation(self):
        frame = Frame(self.ws)
        frame.pack()
        mainmenu = Menu(frame)
        mainmenu.add_command(label="Dodaj dane", command=self.add_model)
        mainmenu.add_command(label="Trenuj model", command=self.train_models)
        mainmenu.add_command(
            label="Identifikuj nagrane próbki", command=self.identify_samples)
        mainmenu.add_command(label="Exit", command=self.ws.destroy)
        self.ws.config(menu=mainmenu)

    def initialize_main_view(self):
        self.initialize_recording_devices()

        label_title = Label(
            self.ws, text="Identyfikacja użytkownika", font='Arial 18')
        label_title.place(relx=0.5, rely=0.10, anchor=CENTER)

        label_info = Label(
            self.ws, text="Wciśnij przycisk i wypowiedź wyświetlone poniżej hasło:", font='Arial 15')
        label_info.place(relx=0.5, rely=0.4, anchor=CENTER)

        label_password = Label(self.ws, text=random.choice(
            PASSWORDS), font='Arial 17 bold')
        label_password.place(relx=0.5, rely=0.6, anchor=CENTER)

        start_btn = Button(
            self.ws,
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
            Radiobutton(self.ws,
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

    def add_model(self):
        print('here will add new model')
        return

    def train_models(self):
        print('here will train models')
        return

    def identify_samples(self):
        print('here will identify samples')
        return

    def record_sample(self):
        print('here will record testing sample')
        return


Identification()
